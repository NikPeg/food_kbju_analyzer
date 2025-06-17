import yaml
import subprocess
import os
import signal
import sys
import socket
import time

class VLLMServer:
    PID_FILE = ".vllm_server.pid"
    LOG_FILE = "vllm_server.log"

    def __init__(self, config_path: str, hf_token: str = ""):
        self.config = self._load_config(config_path)
        self.process = None
        self.hf_token = hf_token
        self._cleanup_old_server()
        self._start_server()

    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _cleanup_old_server(self):
        if os.path.exists(self.PID_FILE):
            try:
                with open(self.PID_FILE, "r") as f:
                    pid = int(f.read())
                os.killpg(pid, signal.SIGINT)
                print(f"Убиваем старый процесс vLLM с PGID={pid}")
            except Exception as e:
                print(f"Не удалось убить старый процесс: {e}")
            os.remove(self.PID_FILE)
            # очищаем старый лог
            if os.path.exists(self.LOG_FILE):
                os.remove(self.LOG_FILE)

    def _wait_for_port(self, host: str, port: int):
        print(f"⏳ Ожидание порта {port} на {host}…")
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                try:
                    sock.connect((host, port))
                    print(f"✅ Порт {port} на {host} открыт — сервер готов.")
                    return
                except (ConnectionRefusedError, socket.timeout):
                    time.sleep(0.5)

    def _start_server(self):
        model = self.config["model"]
        tensor_parallel_size = self.config.get("tensor_parallel_size")
        max_num_seqs = self.config.get("max_num_seqs")
        dtype = self.config.get("dtype")
        port = self.config.get("port")
        host = self.config.get("host")
        served_model_name = self.config.get("served_model_name")
        max_model_len = self.config.get("max_model_len")

        env = os.environ.copy()
        env["HUGGING_FACE_HUB_TOKEN"] = self.hf_token

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--max-num-seqs", str(max_num_seqs),
            "--dtype", dtype,
            "--port", str(port),
            "--host", host,
            "--served-model-name", served_model_name,
        ]
        if  max_model_len:
            cmd.extend(
                [
                    "--max-model-len", str(max_model_len)
                ]
            )
        cmd = [arg for arg in cmd if arg]

        print(f"🚀 Запуск vLLM-сервера на {host}:{port} с моделью {model}…")
        # открываем файл лога для дочернего процесса
        log_f = open(self.LOG_FILE, "a")
        # создаём новую группу процессов, чтобы потом было удобно убить всё вместе
        self.process = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=log_f,
            preexec_fn=os.setsid,
            env=env
        )
        with open(self.PID_FILE, "w") as f:
            f.write(str(self.process.pid))

        # ждём, пока сервер начнёт слушать порт
        self._wait_for_port(host, port)
        print(f"📜 Логи сервера пишутся в файл: {self.LOG_FILE}")

    def stop(self):
        if self.process and self.process.poll() is None:
            pgid = self.process.pid
            print("🛑 Остановка vLLM-сервера (SIGINT группе)…")
            os.killpg(pgid, signal.SIGINT)
            try:
                self.process.wait()
                print("✅ Сервер завершился корректно.")
            except Exception as e:
                print(f"❌ Ошибка при ожидании завершения: {e}")
                print("⚠️ Попытка грубого завершения…")
                os.killpg(pgid, signal.SIGTERM)
                try:
                    self.process.wait()
                except Exception:
                    print("⚠️ Принудительное убийство…")
                    os.killpg(pgid, signal.SIGKILL)

            self.process = None

        if os.path.exists(self.PID_FILE):
            os.remove(self.PID_FILE)

    def __del__(self):
        self.stop()
