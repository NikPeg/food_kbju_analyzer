# БЖУк - Анализатор питательности блюд
Telegram-бот для определения БЖУ (белки, жиры, углеводы) по фото еды с помощью ИИ.

## Зависимости
- Python 3.11+
- Telegram аккаунт

## Быстрый запуск
```bash
curl -L https://disk.yandex.ru/d/HYq-6fk9yI6O3w -o models/best_model.pth
git clone https://github.com/yourusername/nutrition-bot.git
cd nutrition-bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
BOT_TOKEN=ваш_токен_от_BotFather
# USE_CUDA=true  # Для GPU
python bot.py
