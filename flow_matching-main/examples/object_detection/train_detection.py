# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

"""
目标检测Flow Matching训练脚本
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from detection_flow_matching import ObjectDetectionFlowMatching


class COCODetectionDataset(Dataset):
    """
    COCO格式的目标检测数据集
    """
    
    def __init__(
        self,
        data_dir: str,
        annotations_file: str,
        image_size: Tuple[int, int] = (224, 224),
        max_objects: int = 10
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.max_objects = max_objects
        
        # 加载标注文件
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 特征提取器
        self.feature_extractor = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor.eval()
        
        # 冻结特征提取器参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def __len__(self):
        return len(self.annotations['images'])
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取一个样本
        """
        image_info = self.annotations['images'][idx]
        image_id = image_info['id']
        
        # 加载图像
        image_path = os.path.join(self.data_dir, image_info['file_name'])
        image = self._load_image(image_path)
        
        # 提取图像特征
        with torch.no_grad():
            image_tensor = self.transform(image).unsqueeze(0)
            image_features = self.feature_extractor(image_tensor).squeeze()
        
        # 获取该图像的标注
        annotations = [ann for ann in self.annotations['annotations'] if ann['image_id'] == image_id]
        
        # 处理边界框和类别
        bboxes = []
        classes = []
        
        for ann in annotations[:self.max_objects]:  # 限制最大物体数量
            # COCO格式: [x, y, width, height] -> 归一化
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # 归一化到[0,1]
            norm_x = x / image_info['width']
            norm_y = y / image_info['height'] 
            norm_w = w / image_info['width']
            norm_h = h / image_info['height']
            
            bboxes.append([norm_x, norm_y, norm_w, norm_h])
            classes.append(ann['category_id'] - 1)  # COCO类别从1开始，转换为从0开始
        
        # 填充到固定长度
        while len(bboxes) < self.max_objects:
            bboxes.append([0, 0, 0, 0])  # 空边界框
            classes.append(0)  # 背景类别
        
        return {
            'image_features': image_features,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'classes': torch.tensor(classes, dtype=torch.long),
            'image_id': image_id
        }
    
    def _load_image(self, image_path: str):
        """
        加载图像
        """
        from PIL import Image
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个空白图像
            return Image.new('RGB', self.image_size, (0, 0, 0))


class DetectionTrainer:
    """
    目标检测训练器
    """
    
    def __init__(
        self,
        model: ObjectDetectionFlowMatching,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> float:
        """
        训练一个epoch
        """
        self.model.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # 移动到设备
            image_features = batch['image_features'].to(self.device)
            bboxes = batch['bboxes'].to(self.device)
            classes = batch['classes'].to(self.device)
            
            # 处理每个物体
            batch_loss = 0.0
            valid_objects = 0
            
            for i in range(image_features.shape[0]):
                # 找到非空边界框
                valid_mask = (bboxes[i].sum(dim=1) > 0)  # 非零边界框
                
                if valid_mask.sum() > 0:
                    valid_bboxes = bboxes[i][valid_mask]
                    valid_classes = classes[i][valid_mask]
                    valid_features = image_features[i].unsqueeze(0).expand(valid_bboxes.shape[0], -1)
                    
                    # 训练步骤
                    loss_dict = self.model.train_step(
                        image_features=valid_features,
                        target_bboxes=valid_bboxes,
                        target_classes=valid_classes,
                        optimizer=self.optimizer
                    )
                    
                    batch_loss += loss_dict['loss']
                    valid_objects += valid_bboxes.shape[0]
            
            if valid_objects > 0:
                batch_loss /= valid_objects
                total_loss += batch_loss
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
        
        return total_loss / max(num_batches, 1)
    
    def validate(self) -> float:
        """
        验证
        """
        self.model.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                image_features = batch['image_features'].to(self.device)
                bboxes = batch['bboxes'].to(self.device)
                classes = batch['classes'].to(self.device)
                
                batch_loss = 0.0
                valid_objects = 0
                
                for i in range(image_features.shape[0]):
                    valid_mask = (bboxes[i].sum(dim=1) > 0)
                    
                    if valid_mask.sum() > 0:
                        valid_bboxes = bboxes[i][valid_mask]
                        valid_classes = classes[i][valid_mask]
                        valid_features = image_features[i].unsqueeze(0).expand(valid_bboxes.shape[0], -1)
                        
                        # 计算验证损失（不更新参数）
                        batch_size = valid_bboxes.shape[0]
                        x_0 = torch.randn_like(valid_bboxes)
                        x_1 = valid_bboxes
                        t = torch.rand(batch_size, device=self.device)
                        
                        path_sample = self.model.path.sample(x_0=x_0, x_1=x_1, t=t)
                        model_output = self.model.wrapped_model(
                            x=path_sample.x_t,
                            t=t,
                            image_features=valid_features,
                            class_condition=valid_classes
                        )
                        
                        loss = self.model.loss_fn(path_sample=path_sample, model_output=model_output)
                        batch_loss += loss.item()
                        valid_objects += valid_bboxes.shape[0]
                
                if valid_objects > 0:
                    batch_loss /= valid_objects
                    total_loss += batch_loss
                    num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(self, num_epochs: int, save_dir: str = "./checkpoints"):
        """
        完整训练流程
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # 更新学习率
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"Saved best model with val_loss: {val_loss:.4f}")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # 绘制训练曲线
        self.plot_training_curves(save_dir)
    
    def plot_training_curves(self, save_dir: str):
        """
        绘制训练曲线
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss (Log Scale)')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        plt.show()


def create_synthetic_dataset(data_dir: str, num_samples: int = 1000):
    """
    创建合成数据集用于演示
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建合成标注
    annotations = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': i+1, 'name': f'class_{i}'} for i in range(10)
        ]
    }
    
    for i in range(num_samples):
        # 创建合成图像
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image_path = os.path.join(data_dir, f'image_{i:06d}.jpg')
        
        from PIL import Image
        Image.fromarray(image).save(image_path)
        
        # 添加图像信息
        annotations['images'].append({
            'id': i,
            'file_name': f'image_{i:06d}.jpg',
            'width': 224,
            'height': 224
        })
        
        # 添加随机标注
        num_objects = np.random.randint(1, 5)
        for j in range(num_objects):
            x = np.random.randint(0, 150)
            y = np.random.randint(0, 150)
            w = np.random.randint(20, 74)
            h = np.random.randint(20, 74)
            
            annotations['annotations'].append({
                'id': len(annotations['annotations']),
                'image_id': i,
                'category_id': np.random.randint(1, 11),
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0
            })
    
    # 保存标注文件
    with open(os.path.join(data_dir, 'annotations.json'), 'w') as f:
        json.dump(annotations, f)
    
    print(f"Created synthetic dataset with {num_samples} samples in {data_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Object Detection Flow Matching')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--create_synthetic', action='store_true', help='Create synthetic dataset')
    
    args = parser.parse_args()
    
    # 创建合成数据集
    if args.create_synthetic:
        create_synthetic_dataset(args.data_dir, num_samples=1000)
    
    # 创建数据集
    train_dataset = COCODetectionDataset(
        data_dir=args.data_dir,
        annotations_file=os.path.join(args.data_dir, 'annotations.json'),
        max_objects=5
    )
    
    # 分割训练和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    # 初始化模型
    model = ObjectDetectionFlowMatching(
        num_classes=args.num_classes,
        image_feature_dim=2048,
        device=args.device
    )
    
    # 创建训练器
    trainer = DetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device
    )
    
    # 开始训练
    trainer.train(num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()


