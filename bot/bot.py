import asyncio
import logging
import torch
import torchvision.transforms as transforms
import clip
from PIL import Image
import io
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message, BufferedInputFile
from aiogram import F
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import torch

from config import config
from model import create_model

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{config.LOGS_DIR}/bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NutritionPredictor:
    def __init__(self, model_path, device):
        self.device = device
        
        self.model = create_model()
        
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Модель загружена с {model_path}")
            else:
                logger.warning(f"Файл модели {model_path} не найден. Используется неинициализированная модель.")
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель: {e}. Используется неинициализированная модель.")
        
        self.model.to(device)
        self.model.eval()
        
    
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image, text_description=None):
        """Предсказывает БЖУ для изображения с опциональным текстом"""
        try:
        
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
        
            text_tokens = None
            if text_description:
                try:
                    text_tokens = clip.tokenize([text_description], truncate=True).to(self.device)
                except Exception as e:
                    logger.warning(f"Ошибка токенизации текста: {e}")
                    text_tokens = None
            
        
            with torch.no_grad():
                predictions = self.model(input_tensor, text_tokens)
            
        
            protein = predictions['protein'][0].cpu().item()
            fat = predictions['fat'][0].cpu().item()
            carbs = predictions['carbs'][0].cpu().item()
            calories = predictions['calories'][0].cpu().item()
            
            combination_weight = predictions['combination_weight'].cpu().item()
            bzu_proportions = predictions['bzu_proportions'][0].cpu().numpy()
            
            return {
                'protein': max(0, protein),
                'fat': max(0, fat),
                'carbs': max(0, carbs),
                'calories': max(0, calories),
                'combination_weight': combination_weight,
                'proportions': {
                    'protein_prop': bzu_proportions[0],
                    'fat_prop': bzu_proportions[1],
                    'carbs_prop': bzu_proportions[2]
                },
            
            }
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            return None

class NutritionVisualizer:
    @staticmethod
    def create_nutrition_chart(nutrition_data, food_name="Блюдо"):
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        
        protein = nutrition_data['protein']
        fat = nutrition_data['fat'] 
        carbs = nutrition_data['carbs']
        calories = nutrition_data['calories']
        proportions = nutrition_data.get('proportions', {})
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        nutrients = ['Белки', 'Жиры', 'Углеводы']
        values = [protein, fat, carbs]
        
    
        if sum(values) > 0:
            wedges, texts, autotexts = ax1.pie(
                values, 
                labels=nutrients,
                colors=colors,
                autopct='%1.1f г',
                startangle=90,
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax1.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax1.transAxes)
        
        ax1.set_title(f'Состав БЖУ\n{food_name}', fontsize=14, fontweight='bold', pad=20)
        
    
        bars = ax2.bar(nutrients, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}г', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
        
        ax2.set_title('Содержание на 100г', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Граммы', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 10)
        
    
        calorie_breakdown = [protein * 4, fat * 9, carbs * 4]
        if sum(calorie_breakdown) > 0:
            calorie_labels = [f'{nutrients[i]}\n{calorie_breakdown[i]:.0f} ккал' for i in range(3)]
            
            wedges3, texts3, autotexts3 = ax3.pie(
                calorie_breakdown,
                labels=calorie_labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 10, 'fontweight': 'bold'}
            )
            
            for autotext in autotexts3:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
        else:
            ax3.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax3.transAxes)
        
        ax3.set_title(f'Распределение калорий\nВсего: {calories:.0f} ккал', 
                     fontsize=14, fontweight='bold', pad=20)
        
    
        ax4.axis('off')
        
        info_text = f"""
        🍽️ ДЕТАЛЬНЫЙ АНАЛИЗ

        📊 Макронутриенты (на 100г):
        • Белки: {protein:.1f}г ({proportions.get('protein_prop', 0)*100:.1f}%)
        • Жиры: {fat:.1f}г ({proportions.get('fat_prop', 0)*100:.1f}%)
        • Углеводы: {carbs:.1f}г ({proportions.get('carbs_prop', 0)*100:.1f}%)

        ⚡ Энергетическая ценность:
        • Общая калорийность: {calories:.0f} ккал
        • От белков: {protein*4:.0f} ккал
        • От жиров: {fat*9:.0f} ккал  
        • От углеводов: {carbs*4:.0f} ккал


        💡 Рекомендации:
        • {'Сбалансированный состав' if max(proportions.values()) < 0.6 else 'Преобладает один макронутриент'}
        • {'Высококалорийное блюдо' if calories > 300 else 'Среднекалорийное блюдо' if calories > 150 else 'Низкокалорийное блюдо'}

        ⚠️  Данные приблизительные !!!
        """
        
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'🍽️ Полный анализ питательности: {food_name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.05)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close()
        
        return buf

bot = Bot(token=config.BOT_TOKEN)
dp = Dispatcher()

device = torch.device('cuda' if torch.cuda.is_available() and config.DEVICE == 'cuda' else 'cpu')
predictor = NutritionPredictor(config.MODEL_PATH, device)

@dp.message(Command("start"))
async def start_handler(message: Message):
    welcome_text = """
    🍽️ **Добро пожаловать в БЖУк** 

    Я использую продвинутую ИИ-модель для анализа пищевой ценности блюд!

    📸 **Как пользоваться:**
    • Отправьте фото блюда
    • Добавьте описание к фото для повышения точности
    • Получите детальный анализ БЖУ на 100г продукта

    ⚡ **Мои возможности:**
    • Анализ изображения + текста одновременно
    • Определение белков, жиров, углеводов
    • Расчет калорийности и пропорций

    🚀 **Отправьте фото еды с описанием для лучших результатов!**
        """
    
    await message.answer(welcome_text, parse_mode="Markdown")

@dp.message(Command("help"))
async def help_handler(message: Message):
    help_text = """
    📋 **Справка по БЖУк**

    🔹 **/start** - начать работу с ботом
    🔹 **/help** - показать эту справку
    🔹 **/status** - статус системы

    📸 **Отправка фото:**
    • Поддерживаются: JPG, PNG, WEBP
    • Максимальный размер: 20MB
    • Одно блюдо на фото работает лучше

    💬 **Текстовые описания:**
    • "Fried chiken with patatos"
    • "Pizza 4 cheese"

    ⚠️ **Ограничения:**
    • Результаты приблизительные (±15-25%)
    • Не заменяет точные измерения
    • Лучше работает с простыми блюдами
        """
    
    await message.answer(help_text, parse_mode="Markdown")


@dp.message(Command("status"))
async def status_handler(message: Message):
    status_text = f"""
    📊 **Статус системы**

    🔧 **Устройство:** {device}
    💾 **Модель:** {'✅ Загружена' if os.path.exists(config.MODEL_PATH) else '❌ Не найдена'}
    📁 **Конфигурация:** ✅ OK
    🕐 **Время:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    {"🟢 Система готова к работе" if os.path.exists(config.MODEL_PATH) else "Система не готова..."}
        """
    
    await message.answer(status_text, parse_mode="Markdown")

user_requests = {}

async def check_rate_limit(user_id: int) -> bool:
    current_time = datetime.now()
    
    if user_id not in user_requests:
        user_requests[user_id] = []
        user_requests[user_id] = [
        t for t in user_requests[user_id] 
        if (current_time - t).total_seconds() < 60
    ]
    
    if len(user_requests[user_id]) >= config.MAX_REQUESTS_PER_MINUTE:
        return False
    
    user_requests[user_id].append(current_time)
    return True

@dp.message(F.photo)
async def handle_photo(message: Message):
    
    try:
        if not await check_rate_limit(message.from_user.id):
            await message.answer(" Слишком много запросов! Пожалуйста, подождите.")
            return
        
        photo = message.photo[-1]
        if photo.file_size > config.MAX_FILE_SIZE:
            await message.answer(f"❌ Размер файла превышает {config.MAX_FILE_SIZE // 1024 // 1024}MB. Пожалуйста, отправьте фото меньшего размера.")
            return
        processing_msg = await message.answer("🧠 Анализирую фото с помощью ИИ-модели...")
        
        file_info = await bot.get_file(photo.file_id)
        photo_bytes = await bot.download_file(file_info.file_path)
        
        text_description = message.caption if message.caption else None
        food_name = text_description if text_description else "Ваше блюдо"
        
        nutrition_data = predictor.predict(photo_bytes.read(), text_description)
        
        if nutrition_data is None:
            await processing_msg.edit_text("❌ Не удалось обработать изображение. Попробуйте другое фото.")
            return
        
        chart_buffer = NutritionVisualizer.create_nutrition_chart(nutrition_data, food_name)
        
        response_text = f"""
        🍽️ **Анализ блюда: {food_name}**

        📊 **Пищевая ценность на 100г:**
        🥩 Белки: **{nutrition_data['protein']:.1f}г**
        🧈 Жиры: **{nutrition_data['fat']:.1f}г** 
        🍞 Углеводы: **{nutrition_data['carbs']:.1f}г**

        ⚡ **Калорийность: {nutrition_data['calories']:.0f} ккал**


        {'📝 *Использовано описание для повышения точности*' if text_description else '💡 *Добавьте описание к фото для более точного анализа*'}
                """
        
        chart_file = BufferedInputFile(chart_buffer.getvalue(), filename="nutrition_chart.png")
        await message.answer_photo(
            photo=chart_file,
            caption=response_text,
            parse_mode="Markdown"
        )
        
        await processing_msg.delete()
        
    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {e}")
        await message.answer("❌ Произошла ошибка при анализе фото. Попробуйте еще раз.")

@dp.message()
async def handle_text(message: Message):
    await message.answer(
        "📸 Отправьте мне фото блюда для анализа!\n\n"
        "💡 **Совет:** Добавьте описание к фото для повышения точности анализа.",
        reply_to_message_id=message.message_id
    )

async def cleanup_old_requests():
    """Регулярная очистка старых записей"""
    while True:
        await asyncio.sleep(300)  # 5 минут
        current_time = datetime.now()
        for user_id in list(user_requests.keys()):
            # Удаляем записи старше 90  секунд
            user_requests[user_id] = [
                t for t in user_requests[user_id] 
                if (current_time - t).total_seconds() < 90
            ]
            
            if not user_requests[user_id]:
                del user_requests[user_id]

async def main():
    logger.info("Запуск БЖЖЖЖУк...")
    logger.info(f"Устройство: {device}")
    logger.info(f"Модель: {config.MODEL_PATH}")
    logger.info(f"Статус модели: {'Загружена' if os.path.exists(config.MODEL_PATH) else 'Не найдена'}")
    asyncio.create_task(cleanup_old_requests())
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
        
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
    finally:
        await bot.session.close()

if __name__ == "__main__":
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
