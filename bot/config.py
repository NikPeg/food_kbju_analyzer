import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    BOT_TOKEN: str = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
    
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/best_model.pth")
    
    DEVICE: str = "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu"
    
    # Максимальный размер файла 
    MAX_FILE_SIZE: int = 20 * 1024 * 1024  # 20MB
    
    # Лимиты для пользователей
    MAX_REQUESTS_PER_MINUTE: int = 30
    
    # Настройки логирования
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    LOGS_DIR: str = "logs"
    MODELS_DIR: str = "models"

config = Config()

os.makedirs(config.LOGS_DIR, exist_ok=True)
os.makedirs(config.MODELS_DIR, exist_ok=True)
