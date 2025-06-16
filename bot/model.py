"""
Nutrition Model Definition
Модель для предсказания питательной ценности блюд по изображениям с текстом
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import clip
import logging

logger = logging.getLogger(__name__)

class CLIPWithCustomVision(nn.Module):
    """CLIP модель с кастомным ResNet50 визуальным энкодером"""
    
    def __init__(self, clip_model_name='RN50', freeze_text_encoder=False, 
                 text_lr_multiplier=0.1, unfreez_layer=5):
        super().__init__()
        
        # Загружаем CLIP модель
        self.clip_model, self.preprocess = clip.load(clip_model_name, device='cpu')
        
        self.unfreez_layer = unfreez_layer
        self.freeze_text_encoder = freeze_text_encoder
        self.text_lr_multiplier = text_lr_multiplier
        
        # Настройка текстового энкодера
        if not freeze_text_encoder:
            self._setup_text_encoder_training()
        else:
            self._freeze_text_encoder()
        
        # Простой ResNet50 энкодер
        self.custom_vision_encoder = self._create_vision_encoder()
        
        # Размерности
        self.vision_embed_dim = 2048  # ResNet50 output
        self.text_embed_dim = self.clip_model.text_projection.shape[1]
        
        # Проекционные слои
        self.vision_projection = nn.Linear(self.vision_embed_dim, 512)
        self.text_projection_adapter = nn.Linear(self.text_embed_dim, 512)
        
        # Параметр масштабирования для contrastive learning
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))
        
        # Адаптеры для fine-tuning
        self.vision_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Dropout(0.1)
        )
        
        self.text_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Dropout(0.1)
        )
        
    def _setup_text_encoder_training(self):
        """Настройка частичного обучения текстового энкодера"""
        num_layers_to_unfreeze = self.unfreez_layer
        total_layers = len(self.clip_model.transformer.resblocks)
        
        # Размораживаем последние слои
        for i in range(max(0, total_layers - num_layers_to_unfreeze), total_layers):
            for param in self.clip_model.transformer.resblocks[i].parameters():
                param.requires_grad = True
        
        # Размораживаем финальные слои
        for param in self.clip_model.ln_final.parameters():
            param.requires_grad = True
        
        if hasattr(self.clip_model, 'text_projection') and self.clip_model.text_projection is not None:
            self.clip_model.text_projection.requires_grad = True

    def _freeze_text_encoder(self):
        """Замораживает текстовый энкодер"""
        for param in self.clip_model.transformer.parameters():
            param.requires_grad = False
        for param in self.clip_model.ln_final.parameters():
            param.requires_grad = False
        
        if hasattr(self.clip_model, 'text_projection') and self.clip_model.text_projection is not None:
            self.clip_model.text_projection.requires_grad = False

    def _create_vision_encoder(self):
        """Создает ResNet50 визуальный энкодер"""
        backbone = torchvision.models.resnet50(pretrained=False)
        backbone.fc = nn.Identity()  # Убираем классификационный слой
        return backbone
    
    def encode_image(self, image):
        """Кодирует изображение в признаки"""
        return self.custom_vision_encoder(image)
    
    def encode_text(self, text):
        """Кодирует текст в признаки"""
        text = text.to(next(self.clip_model.parameters()).device)
        
        # Tokenization и embedding
        x = self.clip_model.token_embedding(text)
        
        # Positional embedding
        x = x + self.clip_model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Transformer
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)
        
        # Pooling - берем признаки EOS токена
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        
        # Проекция
        if hasattr(self.clip_model, 'text_projection') and self.clip_model.text_projection is not None:
            x = x @ self.clip_model.text_projection
        
        return x
    
    def forward(self, image, text=None):
        """Прямой проход модели"""
        # Кодируем изображение
        image_features = self.encode_image(image)
        image_features = self.vision_projection(image_features)
        
        # Применяем адаптер к визуальным признакам
        image_features_adapted = image_features + self.vision_adapter(image_features)
        
        if text is not None:
            # Кодируем текст
            text_features = self.encode_text(text)
            text_features = self.text_projection_adapter(text_features)
            
            # Применяем адаптер к текстовым признакам
            text_features_adapted = text_features + self.text_adapter(text_features)
            
            # Нормализация для contrastive learning
            image_features_norm = image_features_adapted / image_features_adapted.norm(dim=1, keepdim=True)
            text_features_norm = text_features_adapted / text_features_adapted.norm(dim=1, keepdim=True)
            
            # Вычисляем logits для CLIP loss
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features_norm @ text_features_norm.t()
            logits_per_text = logits_per_image.t()
            
            return {
                'image_features': image_features_adapted,
                'text_features': text_features_adapted,
                'image_features_norm': image_features_norm,
                'text_features_norm': text_features_norm,
                'logits_per_image': logits_per_image,
                'logits_per_text': logits_per_text,
                'logit_scale': logit_scale
            }
        else:
            # Только изображение
            image_features_norm = image_features_adapted / image_features_adapted.norm(dim=1, keepdim=True)
            return {
                'image_features': image_features_adapted,
                'image_features_norm': image_features_norm
            }


class CrossModalAttention(nn.Module):
    """Cross-modal attention для взаимодействия визуальных и текстовых признаков"""
    
    def __init__(self, visual_dim, text_dim, attention_dim=256):
        super().__init__()
        
        # Проекционные слои
        self.visual_proj = nn.Linear(visual_dim, attention_dim)
        self.text_proj = nn.Linear(text_dim, attention_dim)
        
        # Cross-attention слои
        self.visual_to_text_attn = nn.MultiheadAttention(
            embed_dim=attention_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        self.text_to_visual_attn = nn.MultiheadAttention(
            embed_dim=attention_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Выходные проекции
        self.visual_out = nn.Linear(attention_dim, visual_dim)
        self.text_out = nn.Linear(attention_dim, text_dim)
        
        # Layer normalization
        self.visual_ln = nn.LayerNorm(visual_dim)
        self.text_ln = nn.LayerNorm(text_dim)
        
    def forward(self, visual_features, text_features):
        """Применяет cross-modal attention"""
        # Проецируем в общее пространство attention
        visual_proj = self.visual_proj(visual_features).unsqueeze(1)  # (B, 1, attention_dim)
        text_proj = self.text_proj(text_features).unsqueeze(1)        # (B, 1, attention_dim)
        
        # Visual attending to text
        visual_attended, _ = self.visual_to_text_attn(visual_proj, text_proj, text_proj)
        visual_attended = visual_attended.squeeze(1)
        
        # Text attending to visual
        text_attended, _ = self.text_to_visual_attn(text_proj, visual_proj, visual_proj)
        text_attended = text_attended.squeeze(1)
        
        # Residual connections + layer norm
        visual_out = self.visual_ln(visual_features + self.visual_out(visual_attended))
        text_out = self.text_ln(text_features + self.text_out(text_attended))
        
        return visual_out, text_out


class SimpleNutritionHead(nn.Module):
    """Голова для предсказания питательных веществ с опциональным self-attention"""
    
    def __init__(self, in_features, out_features, dropout=0.3, use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Self-attention блок
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=in_features,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(in_features)
        
        # Основная сеть предсказания
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(in_features // 2, in_features // 4),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(in_features // 4, out_features),
            nn.ReLU()  # Гарантируем неотрицательные значения
        )
    
    def forward(self, x):
        if self.use_attention:
            # Применяем self-attention
            x_seq = x.unsqueeze(1)  # (batch, 1, features)
            attended, _ = self.attention(x_seq, x_seq, x_seq)
            x = self.attention_norm(x + attended.squeeze(1))
        
        return self.net(x)


class EnhancedFoodNutritionNetWithCLIP(nn.Module):
    """Основная модель для предсказания питательной ценности блюд"""
    
    def __init__(self, clip_model_name='RN50', use_nutrition_attention=True, 
                 use_cross_modal_attention=True, dropout=0.2, use_text_features=True, 
                 freeze_text_encoder=False, text_lr_multiplier=0.1):
        super().__init__()
        
        # CLIP модель с кастомным визуальным энкодером
        self.clip_model = CLIPWithCustomVision(
            clip_model_name, 
            freeze_text_encoder=freeze_text_encoder,
            text_lr_multiplier=text_lr_multiplier
        )
        
        # Флаги конфигурации
        self.use_text_features = use_text_features
        self.use_cross_modal_attention = use_cross_modal_attention
        self.use_nutrition_attention = use_nutrition_attention
        
        feature_dim = 512
        
        # Cross-modal attention для взаимодействия модальностей
        if use_cross_modal_attention and use_text_features:
            self.cross_modal_attention = CrossModalAttention(
                visual_dim=feature_dim, 
                text_dim=feature_dim, 
                attention_dim=256
            )
        
        # Fusion слой для объединения визуальных и текстовых признаков
        if use_text_features:
            fusion_features = feature_dim * 2
            self.feature_fusion = nn.Sequential(
                nn.Linear(fusion_features, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.SiLU(),
                nn.Dropout(dropout * 0.5)
            )
        
        # Общие полносвязные слои
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(512, 256),
            nn.SiLU(),
        )
        
        # Головы для предсказания БЖУ
        self.bzu_head = SimpleNutritionHead(256, 3, dropout, use_nutrition_attention)
        self.protein_head = SimpleNutritionHead(256, 1, dropout, use_nutrition_attention)
        self.fat_head = SimpleNutritionHead(256, 1, dropout, use_nutrition_attention)
        self.carb_head = SimpleNutritionHead(256, 1, dropout, use_nutrition_attention)
        
        # Параметр для взвешенного объединения общих и специфичных предсказаний
        self.combination_weight = nn.Parameter(torch.tensor(0.7))
    
    def forward(self, x, text=None):
        """Прямой проход модели"""
        # Получаем признаки от CLIP модели
        clip_output = self.clip_model(x, text)
        
        visual_features = clip_output['image_features']
        
        # Обработка текстовых признаков если доступны
        if self.use_text_features and text is not None and 'text_features' in clip_output:
            text_features = clip_output['text_features']
            
            # Применяем cross-modal attention
            if self.use_cross_modal_attention:
                visual_features, text_features = self.cross_modal_attention(
                    visual_features, text_features
                )
            
            # Объединяем визуальные и текстовые признаки
            features = torch.cat([visual_features, text_features], dim=1)
            features = self.feature_fusion(features)
        else:
            features = visual_features
        
        # Общие признаки через shared FC
        shared = self.shared_fc(features)
        
        # Предсказания от разных голов
        bzu_general = self.bzu_head(shared)
        protein_specific = self.protein_head(shared)
        fat_specific = self.fat_head(shared)
        carb_specific = self.carb_head(shared)
        
        # Взвешенное объединение общих и специфичных предсказаний
        w = torch.sigmoid(self.combination_weight)
        bzu_specific = torch.cat([protein_specific, fat_specific, carb_specific], dim=1)
        bzu = w * bzu_general + (1 - w) * bzu_specific
        
        # Гарантируем неотрицательные значения
        bzu = F.relu(bzu) + 1e-6
        
        # Вычисляем пропорции БЖУ
        bzu_sum = bzu.sum(dim=1, keepdim=True) + 1e-8
        bzu_proportions = bzu / bzu_sum
        
        # Вычисляем калории (4*белки + 9*жиры + 4*углеводы)
        calculated_cal = 4 * bzu[:, 0] + 9 * bzu[:, 1] + 4 * bzu[:, 2]
        
        # Формируем результат
        result = {
            'bzu': bzu,
            'protein': bzu[:, 0],
            'fat': bzu[:, 1], 
            'carbs': bzu[:, 2],
            'calories': calculated_cal,
            'bzu_general': bzu_general,
            'bzu_specific': bzu_specific,
            'calculated_cal': calculated_cal,
            'bzu_proportions': bzu_proportions,
            'combination_weight': w,
        }
        
        # Добавляем CLIP-специфичные выходы если доступны
        if text is not None and 'logits_per_image' in clip_output:
            result.update({
                'clip_logits_per_image': clip_output['logits_per_image'],
                'clip_logits_per_text': clip_output['logits_per_text'],
                'clip_image_features': clip_output['image_features'],
                'clip_text_features': clip_output['text_features'],
            })
        
        return result


def create_model():
    """Создает модель с настройками по умолчанию для загрузки весов"""
    model = EnhancedFoodNutritionNetWithCLIP(
        clip_model_name='RN50',
        use_nutrition_attention=True,
        use_cross_modal_attention=True,
        dropout=0.2,
        use_text_features=True,
        freeze_text_encoder=False,
        text_lr_multiplier=0.1
    )
    return model


def load_model(model_path, device='cpu'):
    """
    Загружает предобученную модель из файла
    
    Args:
        model_path: Путь к файлу модели (например, 'best_chek.pth')
        device: Устройство для загрузки
    
    Returns:
        EnhancedFoodNutritionNetWithCLIP: Загруженная модель
    """
    # Создаем пустую модель
    model = create_model()
    
    try:
        # Загружаем checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Извлекаем state_dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logger.info(f"Загружена модель из checkpoint с эпохи {checkpoint.get('epoch', 'неизвестно')}")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Загружаем веса в модель
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Модель успешно загружена из {model_path}")
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели из {model_path}: {e}")
        logger.warning("Возвращается неинициализированная модель")
    
    model.to(device)
    model.eval()
    return model
