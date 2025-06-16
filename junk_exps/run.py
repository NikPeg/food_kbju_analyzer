
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torchvision.transforms as T
from PIL import ImageFilter
import random

class GaussianBlur:
    def __init__(self, radius=[0.1, 2.0]):
        self.radius = radius
    
    def __call__(self, img):
        radius = random.uniform(self.radius[0], self.radius[1])
        return img.filter(ImageFilter.GaussianBlur(radius))

def fix_path_separators(path, base_path=""):
    """Исправляет разделители путей для текущей ОС"""
    if path:
        
        path = path.replace('\\', os.sep).replace('/', os.sep)
        if base_path:
            return os.path.join(base_path, path)
    return path

def load_json_dataset(json_path):
    """Загружает датасет из JSON файла"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_text_descriptions_from_title(title):
    """Создает различные текстовые описания на основе названия блюда"""
    if not title:
        return ["food dish"]
    
    title = title.lower().strip()
    
    
    descriptions = [title]
    
    
    clean_title = title.replace(" recipe", "").replace("recipe ", "")
    clean_title = clean_title.replace("how to make ", "").replace("homemade ", "")
    
    if clean_title != title:
        descriptions.append(clean_title)
    
    
    food_variations = [
        f"delicious {clean_title}",
        f"homemade {clean_title}",
        f"fresh {clean_title}",
        f"tasty {clean_title}",
        f"healthy {clean_title}",
        f"cooked {clean_title}",
        f"prepared {clean_title}",
        f"natural {clean_title}"
    ]
    
    descriptions.extend(food_variations)
    
    
    keywords_descriptions = {
        'cake': ['sweet cake', 'dessert cake', 'baked cake'],
        'bread': ['fresh bread', 'baked bread', 'soft bread'],
        'soup': ['hot soup', 'warm soup', 'liquid soup'],
        'salad': ['fresh salad', 'green salad', 'mixed salad'],
        'chicken': ['cooked chicken', 'grilled chicken', 'roasted chicken'],
        'beef': ['grilled beef', 'cooked beef', 'roasted beef'],
        'fish': ['cooked fish', 'grilled fish', 'fresh fish'],
        'pasta': ['cooked pasta', 'italian pasta', 'boiled pasta'],
        'rice': ['cooked rice', 'steamed rice', 'white rice'],
        'pizza': ['baked pizza', 'italian pizza', 'cheese pizza'],
        'sandwich': ['fresh sandwich', 'toasted sandwich', 'filled sandwich'],
        'smoothie': ['fruit smoothie', 'healthy smoothie', 'blended smoothie'],
        'juice': ['fresh juice', 'fruit juice', 'natural juice'],
        'tea': ['hot tea', 'herbal tea', 'brewed tea'],
        'coffee': ['hot coffee', 'brewed coffee', 'fresh coffee'],
        'ice cream': ['frozen ice cream', 'cold ice cream', 'creamy ice cream']
    }
    
    for keyword, desc_list in keywords_descriptions.items():
        if keyword in clean_title:
            descriptions.extend(desc_list)
    
    return descriptions

def process_json_to_dataframe(data, base_image_path=""):
    """Преобразует JSON данные в DataFrame"""
    processed_data = []
    
    for item in data:
        
        if len(item.get('images', [])) == 0:
            continue
            
        
        nutr_per100g = item.get('nutr_per100g', {})
        target_bzu = [
            nutr_per100g.get('protein_g', 0.0),
            nutr_per100g.get('fat_g', 0.0),
            nutr_per100g.get('carbs_g', 0.0)
        ]
        
        
        image_info = item['images'][0]
        image_path = image_info.get('valid_path', '')
        
        
        image_path = fix_path_separators(image_path, base_image_path)
        
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        
        text_descriptions = create_text_descriptions_from_title(item.get('title', ''))
        
        processed_item = {
            'id': item['id'],
            'image_path': image_path,
            'target_bzu': target_bzu,
            'target_energy': nutr_per100g.get('energy_kcal', 0.0),
            'text_description': np.random.choice(text_descriptions) if text_descriptions else '',
            'partition': item.get('partition', 'train'),
            'title': item.get('title', ''),
            'protein': target_bzu[0],
            'fat': target_bzu[1], 
            'carbs': target_bzu[2],
            'calories': nutr_per100g.get('energy_kcal', 0.0)
        }
        
        processed_data.append(processed_item)
    
    return pd.DataFrame(processed_data)

class SimpleNutritionDataset(Dataset):
    def __init__(self, dataframe, transform=None, mode='train', use_text=True):
        self.df = dataframe
        self.transform = transform or self._default_transforms(mode)
        self.use_text = use_text
        
    def _default_transforms(self, mode):
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        if mode == 'train':
            return T.Compose([
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                normalize
            ])
        else:
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
            
            image = Image.open(row['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {row['image_path']}: {e}")
            
            image = torch.zeros(3, 224, 224)
            
        
        targets = torch.tensor(row['target_bzu'], dtype=torch.float32)
        
        sample = {
            'image': image,
            'targets': targets,  
            'item_id': row['id'],
            'calories': torch.tensor(row['target_energy'], dtype=torch.float32)
        }
        
        if self.use_text:
            sample['text'] = row['text_description']
            
        return sample

def simple_collate_fn(batch):
    """Простая функция для батчинга"""
    images = torch.stack([item['image'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    calories = torch.stack([item['calories'] for item in batch])
    
    batch_data = {
        'images': images,
        'targets': targets,  
        'calories': calories,
        'item_ids': [item['item_id'] for item in batch]
    }
    
    if 'text' in batch[0]:
        batch_data['texts'] = [item['text'] for item in batch]
    
    return batch_data

def create_simple_dataloader(json_path, batch_size=32, shuffle=True, 
                           transform=None, partition=None, base_img_path=''):
    """Создает простой DataLoader для SimpleNutritionNet"""
    
    print(f"Loading dataset from: {json_path}")
    data = load_json_dataset(json_path)
    print(f"Loaded {len(data)} items")
    
    print(f"Processing data with base path: {base_img_path}")
    df = process_json_to_dataframe(data, base_img_path)
    print(f"Processed {len(df)} valid items")
    
    if partition:
        df = df[df['partition'] == partition]
        print(f"Filtered to partition '{partition}': {len(df)} items")
        
    if len(df) == 0:
        raise ValueError("No data found after filtering!")
    
    
    print("Checking first few image paths:")
    for i, path in enumerate(df['image_path'].head(3)):
        exists = os.path.exists(path)
        print(f"  {i+1}. {path} - {'EXISTS' if exists else 'NOT FOUND'}")
        
    dataset = SimpleNutritionDataset(
        df, 
        transform=transform, 
        mode=partition if partition else 'train'
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=simple_collate_fn,
        pin_memory=True,
        num_workers=0  
    )


def analyze_dataset(json_path, base_img_path=''):
    """Анализирует датасет"""
    data = load_json_dataset(json_path)
    df = process_json_to_dataframe(data, base_img_path)
    
    print("=== Dataset Analysis ===")
    print(f"Total items: {len(df)}")
    
    if 'partition' in df.columns:
        print(f"Partitions: {df['partition'].value_counts().to_dict()}")
    
    print(f"\nNutrition stats:")
    print(f"Protein: mean={df['protein'].mean():.2f}, std={df['protein'].std():.2f}")
    print(f"Fat: mean={df['fat'].mean():.2f}, std={df['fat'].std():.2f}")
    print(f"Carbs: mean={df['carbs'].mean():.2f}, std={df['carbs'].std():.2f}")
    print(f"Calories: mean={df['calories'].mean():.2f}, std={df['calories'].std():.2f}")
    
    
    existing_images = df['image_path'].apply(os.path.exists).sum()
    print(f"\nImage files: {existing_images}/{len(df)} exist")
    
    return df

import torch
import torch.nn as nn
import torchvision.models as models

class SimpleNutritionNet(nn.Module):
    def __init__(self, dropout=0.2, pretrained=True):
        super().__init__()
        
        
        self.protein_net = self._create_resnet_branch(pretrained, dropout)
        self.fat_net = self._create_resnet_branch(pretrained, dropout)
        self.carbs_net = self._create_resnet_branch(pretrained, dropout)
        
    def _create_resnet_branch(self, pretrained, dropout):
        """Создает одну ветку ResNet50 для предсказания одного макронутриента"""
        
        model_path = 'model_e500_v-8.950.pth'
        checkpoint = torch.load(model_path, map_location='cpu', encoding='latin1',weights_only=False)
        resnet = models.resnet50(pretrained=False)  
             
        if isinstance(checkpoint, dict):
            
            resnet_weights = {}
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            for key in state_dict:
                if 'visionMLP' in key or 'visual_embedding' in key:
                    new_key = key.replace('visionMLP.', '').replace('visual_embedding.', '')
                    resnet_weights[new_key] = state_dict[key]
            
            
            resnet.load_state_dict(resnet_weights, strict=False)
        
 
        
        features = nn.Sequential(*list(resnet.children())[:-1])
        
        
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(512, 1),  
            nn.ReLU()  
        )
        
        return nn.Sequential(features, classifier)
    
    def forward(self, image):
        
        protein = self.protein_net(image).squeeze(-1)  
        fat = self.fat_net(image).squeeze(-1)         
        carbs = self.carbs_net(image).squeeze(-1)     
        
        
        bzu = torch.stack([protein, fat, carbs], dim=1)  
        
        
        calories = 4 * protein + 9 * fat + 4 * carbs
        
        return {
            'bzu': bzu,           
            'protein': protein,   
            'fat': fat,          
            'carbs': carbs,      
            'calories': calories  
        }
    
    def get_parameter_groups(self, base_lr=1e-4):
        """Создает группы параметров для разных learning rates"""
        return [
            {
                'params': list(self.protein_net.parameters()),
                'lr': base_lr,
                'name': 'protein_branch'
            },
            {
                'params': list(self.fat_net.parameters()),
                'lr': base_lr,
                'name': 'fat_branch'
            },
            {
                'params': list(self.carbs_net.parameters()),
                'lr': base_lr,
                'name': 'carbs_branch'
            }
        ]


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

class NutritionLoss(nn.Module):
    def __init__(self, 
                 weights=[1.0, 1.0, 1.0],  
                 loss_type='huber',  
                 huber_delta=1.0,
                 calorie_weight=0.1):  
        super().__init__()
        self.weights = weights
        self.calorie_weight = calorie_weight
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss(delta=huber_delta)
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, predictions, batch_data):
        """
        predictions: dict с ключами 'protein', 'fat', 'carbs', 'calories'
        batch_data: dict с ключами 'targets', 'calories' из DataLoader
        """
        
        targets = batch_data['targets']  
        true_calories = batch_data['calories']  
        
        
        protein_loss = self.criterion(predictions['protein'], targets[:, 0])
        fat_loss = self.criterion(predictions['fat'], targets[:, 1])
        carbs_loss = self.criterion(predictions['carbs'], targets[:, 2])
        
        
        calorie_loss = 0
        if self.calorie_weight > 0:
            calorie_loss = self.criterion(predictions['calories'], true_calories)
        
        
        total_loss = (self.weights[0] * protein_loss + 
                     self.weights[1] * fat_loss + 
                     self.weights[2] * carbs_loss +
                     self.calorie_weight * calorie_loss)
        
        return {
            'total_loss': total_loss,
            'protein_loss': protein_loss,
            'fat_loss': fat_loss, 
            'carbs_loss': carbs_loss,
            'calorie_loss': calorie_loss
        }

def calculate_metrics(predictions, batch_data):
    """Вычисляет метрики качества предсказаний"""
    metrics = {}
    
    
    targets = batch_data['targets']  
    true_calories = batch_data['calories']  
    
    
    true_values = {
        'protein': targets[:, 0].cpu().numpy(),
        'fat': targets[:, 1].cpu().numpy(),
        'carbs': targets[:, 2].cpu().numpy(),
        'calories': true_calories.cpu().numpy()
    }
    
    for nutrient in ['protein', 'fat', 'carbs', 'calories']:
        y_true = true_values[nutrient]
        y_pred = predictions[nutrient].cpu().numpy()
        
        metrics[f'{nutrient}_mae'] = mean_absolute_error(y_true, y_pred)
        metrics[f'{nutrient}_mse'] = mean_squared_error(y_true, y_pred)
        metrics[f'{nutrient}_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        metrics[f'{nutrient}_mape'] = mape
    
    return metrics

class IndependentNutritionTrainer:
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 loss_fn,
                 optimizer,
                 scheduler=None,
                 device='cuda',
                 save_dir='./checkpoints',
                 use_wandb=False):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        
        self.best_val_loss = float('inf')
        self.best_individual_losses = {'protein': float('inf'), 'fat': float('inf'), 'carbs': float('inf')}
        self.train_losses = []
        self.val_losses = []
        self.individual_train_losses = {'protein': [], 'fat': [], 'carbs': []}
        self.individual_val_losses = {'protein': [], 'fat': [], 'carbs': []}
        
        os.makedirs(save_dir, exist_ok=True)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_losses = {'total_loss': [], 'protein_loss': [], 'fat_loss': [], 'carbs_loss': []}
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch_data in enumerate(pbar):
            images = batch_data['images'].to(self.device)
            targets = batch_data['targets'].to(self.device)
            calories = batch_data['calories'].to(self.device)
            
            batch_data_device = {
                'targets': targets,
                'calories': calories,
                'item_ids': batch_data['item_ids']
            }
            
            self.optimizer.zero_grad()
            
            
            predictions = self.model(images)
            
            
            loss_dict = self.loss_fn(predictions, batch_data_device)
            loss = loss_dict['total_loss']
            
            
            loss.backward()
            
            
            torch.nn.utils.clip_grad_norm_(self.model.protein_net.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.fat_net.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.carbs_net.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            
            total_loss += loss.item()
            for key in all_losses:
                if key in loss_dict:
                    all_losses[key].append(loss_dict[key].item())
            
            
            pbar.set_postfix({
                'total': f'{loss.item():.4f}',
                'P': f'{loss_dict["protein_loss"].item():.3f}',
                'F': f'{loss_dict["fat_loss"].item():.3f}',
                'C': f'{loss_dict["carbs_loss"].item():.3f}'
            })
        
        avg_losses = {k: np.mean(v) for k, v in all_losses.items()}
        return avg_losses
    
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        all_losses = {'total_loss': [], 'protein_loss': [], 'fat_loss': [], 'carbs_loss': []}
        all_predictions = {'protein': [], 'fat': [], 'carbs': [], 'calories': []}
        all_batch_data = {'targets': [], 'calories': []}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch_data in pbar:
                images = batch_data['images'].to(self.device)
                targets = batch_data['targets'].to(self.device)
                calories = batch_data['calories'].to(self.device)
                
                batch_data_device = {
                    'targets': targets,
                    'calories': calories,
                    'item_ids': batch_data['item_ids']
                }
                
                
                predictions = self.model(images)
                
                
                loss_dict = self.loss_fn(predictions, batch_data_device)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                for key in all_losses:
                    if key in loss_dict:
                        all_losses[key].append(loss_dict[key].item())
                
                
                for nutrient in all_predictions:
                    all_predictions[nutrient].append(predictions[nutrient])
                
                all_batch_data['targets'].append(targets)
                all_batch_data['calories'].append(calories)
                
                pbar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'P': f'{loss_dict["protein_loss"].item():.3f}',
                    'F': f'{loss_dict["fat_loss"].item():.3f}',
                    'C': f'{loss_dict["carbs_loss"].item():.3f}'
                })
        
        
        final_predictions = {}
        for nutrient in all_predictions:
            final_predictions[nutrient] = torch.cat(all_predictions[nutrient])
        
        final_batch_data = {
            'targets': torch.cat(all_batch_data['targets']),
            'calories': torch.cat(all_batch_data['calories'])
        }
        
        
        metrics = calculate_metrics(final_predictions, final_batch_data)
        avg_losses = {k: np.mean(v) for k, v in all_losses.items()}
        
        return avg_losses, metrics
    
    def train(self, num_epochs, save_every=5):
        print(f"🚀 Начинаем обучение 3 независимых ResNet50 на {num_epochs} эпох...")
        print(f"📱 Устройство: {self.device}")
        print(f"📊 Train dataset: {len(self.train_loader.dataset)} образцов")
        print(f"📊 Val dataset: {len(self.val_loader.dataset)} образцов")
        print(f"🎯 Модель: 3 независимые сети для белков, жиров и углеводов")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            
            train_losses = self.train_epoch()
            
            
            val_losses, val_metrics = self.validate_epoch()
            
            
            self.train_losses.append(train_losses['total_loss'])
            self.val_losses.append(val_losses['total_loss'])
            
            
            for nutrient in ['protein', 'fat', 'carbs']:
                self.individual_train_losses[nutrient].append(train_losses[f'{nutrient}_loss'])
                self.individual_val_losses[nutrient].append(val_losses[f'{nutrient}_loss'])
            
            
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total_loss'])
                else:
                    self.scheduler.step()
            
            
            is_best_total = val_losses['total_loss'] < self.best_val_loss
            if is_best_total:
                self.best_val_loss = val_losses['total_loss']
                self.save_checkpoint(epoch, is_best=True, checkpoint_type='best_total')
                print(f"🎉 Новая лучшая ОБЩАЯ модель! Val loss: {self.best_val_loss:.4f}")
            
            
            for nutrient in ['protein', 'fat', 'carbs']:
                current_loss = val_losses[f'{nutrient}_loss']
                if current_loss < self.best_individual_losses[nutrient]:
                    self.best_individual_losses[nutrient] = current_loss
                    self.save_checkpoint(epoch, is_best=True, checkpoint_type=f'best_{nutrient}')
                    print(f"🎯 Лучшая модель для {nutrient.upper()}! Loss: {current_loss:.4f}")
            
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch)
            
            
            print(f"\n📈 РЕЗУЛЬТАТЫ ЭПОХИ {epoch+1}:")
            print(f"Train Total Loss: {train_losses['total_loss']:.4f}")
            print(f"  ├─ Protein: {train_losses['protein_loss']:.4f}")
            print(f"  ├─ Fat:     {train_losses['fat_loss']:.4f}")
            print(f"  └─ Carbs:   {train_losses['carbs_loss']:.4f}")
            
            print(f"Val Total Loss: {val_losses['total_loss']:.4f}")
            print(f"  ├─ Protein: {val_losses['protein_loss']:.4f}")
            print(f"  ├─ Fat:     {val_losses['fat_loss']:.4f}")
            print(f"  └─ Carbs:   {val_losses['carbs_loss']:.4f}")
            
            print(f"\n📊 МЕТРИКИ ВАЛИДАЦИИ:")
            for nutrient in ['protein', 'fat', 'carbs', 'calories']:
                mae = val_metrics[f'{nutrient}_mae']
                mape = val_metrics[f'{nutrient}_mape']
                print(f"  {nutrient.upper():8}: MAE={mae:5.2f}g, MAPE={mape:5.1f}%")
            
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n⚙️  Learning Rate: {current_lr:.2e}")
            
            
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train_total_loss': train_losses['total_loss'],
                    'val_total_loss': val_losses['total_loss'],
                    'lr': current_lr
                }
                
                
                for nutrient in ['protein', 'fat', 'carbs']:
                    log_dict[f'train_{nutrient}_loss'] = train_losses[f'{nutrient}_loss']
                    log_dict[f'val_{nutrient}_loss'] = val_losses[f'{nutrient}_loss']
                
                log_dict.update(val_metrics)
                wandb.log(log_dict)
        
        print(f"\n🎊 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"🏆 Лучший общий validation loss: {self.best_val_loss:.4f}")
        print(f"🎯 Лучшие индивидуальные потери:")
        for nutrient in ['protein', 'fat', 'carbs']:
            print(f"   {nutrient.upper()}: {self.best_individual_losses[nutrient]:.4f}")
    
    def save_checkpoint(self, epoch, is_best=False, checkpoint_type='regular'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_individual_losses': self.best_individual_losses,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'individual_train_losses': self.individual_train_losses,
            'individual_val_losses': self.individual_val_losses
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            filename = f'{checkpoint_type}.pth'
        else:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"💾 Сохранен чекпоинт: {filename}")
    
    def plot_losses(self):
        if len(self.train_losses) == 0:
            print("Нет данных для построения графика")
            return
            
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train Total', alpha=0.8, linewidth=2)
        plt.plot(self.val_losses, label='Val Total', alpha=0.8, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Общие потери (сумма всех нутриентов)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        nutrients = ['protein', 'fat', 'carbs']
        colors = ['red', 'green', 'blue']
        
        for i, (nutrient, color) in enumerate(zip(nutrients, colors)):
            plt.subplot(2, 2, i+2)
            plt.plot(self.individual_train_losses[nutrient], 
                    label=f'Train {nutrient.capitalize()}', 
                    alpha=0.8, linewidth=2, color=color)
            plt.plot(self.individual_val_losses[nutrient], 
                    label=f'Val {nutrient.capitalize()}', 
                    alpha=0.8, linewidth=2, color=color, linestyle='--')
            plt.xlabel('Epoch')
            plt.ylabel(f'{nutrient.capitalize()} Loss')
            plt.title(f'Потери для {nutrient.upper()} (независимая сеть)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/independent_training_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        for nutrient, color in zip(nutrients, colors):
            plt.plot(self.individual_train_losses[nutrient], 
                    label=f'Train {nutrient.capitalize()}', 
                    alpha=0.7, color=color)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Сравнение Train потерь по нутриентам')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for nutrient, color in zip(nutrients, colors):
            plt.plot(self.individual_val_losses[nutrient], 
                    label=f'Val {nutrient.capitalize()}', 
                    alpha=0.7, color=color, linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Сравнение Val потерь по нутриентам')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/nutrients_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


    
train_loader = create_simple_dataloader(
    'train.json',
    batch_size=64,
    partition='train',
    base_img_path='/home/student/minazarko/cv/train',
    transform=  T.Compose([
                T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=0.3),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.2, scale=(0.02, 0.1)), 
            ])
)

val_loader = create_simple_dataloader(
    'val.json',
    batch_size=64,
    partition='val',
    base_img_path='/home/student/minazarko/cv/val_valid',
   
)
       
model = SimpleNutritionNet(dropout=0.2, pretrained=True)
    

loss_fn = NutritionLoss(
    weights=[1.0, 1.2, 1.0], 
    loss_type='huber',
    calorie_weight=0.1
)


param_groups = model.get_parameter_groups(base_lr=1e-4)
optimizer = optim.AdamW(param_groups, weight_decay=1e-4)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

trainer = IndependentNutritionTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='./nutrition_checkpoints',
    use_wandb=False 
)
    

trainer.train(num_epochs=50, save_every=10)


trainer.plot_losses()
