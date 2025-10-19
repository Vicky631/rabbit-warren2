# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

"""
目标检测Flow Matching推理和可视化脚本
"""

import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any
import json

from detection_flow_matching import ObjectDetectionFlowMatching


class DetectionInference:
    """
    目标检测推理类
    """
    
    def __init__(
        self,
        model_path: str,
        num_classes: int = 80,
        device: str = "cuda"
    ):
        self.device = device
        self.num_classes = num_classes
        
        # 初始化模型
        self.model = ObjectDetectionFlowMatching(
            num_classes=num_classes,
            image_feature_dim=2048,
            device=device
        )
        
        # 加载预训练权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using random weights.")
        
        # 特征提取器
        self.feature_extractor = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor.eval()
        self.feature_extractor.to(device)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # COCO类别名称（简化版）
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def extract_image_features(self, image: Image.Image) -> torch.Tensor:
        """
        提取图像特征
        """
        with torch.no_grad():
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            features = self.feature_extractor(image_tensor).squeeze()
        return features
    
    def detect_objects_in_image(
        self,
        image_path: str,
        target_classes: List[int] = None,
        confidence_threshold: float = 0.3,
        num_samples: int = 50,
        num_steps: int = 100
    ) -> Dict[str, Any]:
        """
        在图像中检测指定类别的物体
        
        Args:
            image_path: 图像路径
            target_classes: 要检测的类别列表，None表示检测所有类别
            confidence_threshold: 置信度阈值
            num_samples: 每个类别生成的样本数
            num_steps: ODE求解步数
            
        Returns:
            检测结果字典
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # 提取特征
        image_features = self.extract_image_features(image)
        
        all_detections = []
        
        # 确定要检测的类别
        if target_classes is None:
            target_classes = list(range(self.num_classes))
        
        # 对每个目标类别进行检测
        for class_id in target_classes:
            print(f"检测类别 {class_id}: {self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'}")
            
            # 生成检测结果
            bboxes, confidences = self.model.detect_objects(
                image_features=image_features.unsqueeze(0),
                target_class=class_id,
                num_samples=num_samples,
                num_steps=num_steps
            )
            
            # 后处理
            final_bboxes, final_confidences = self.model.post_process_detections(
                bboxes, confidences, 
                confidence_threshold=confidence_threshold
            )
            
            # 转换回原始图像坐标
            if len(final_bboxes) > 0:
                # 从归一化坐标转换回像素坐标
                final_bboxes[:, 0] *= original_size[0]  # x
                final_bboxes[:, 1] *= original_size[1]  # y
                final_bboxes[:, 2] *= original_size[0]  # width
                final_bboxes[:, 3] *= original_size[1]  # height
                
                for bbox, conf in zip(final_bboxes, final_confidences):
                    all_detections.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}',
                        'bbox': bbox.cpu().numpy(),
                        'confidence': conf.item()
                    })
        
        return {
            'image_path': image_path,
            'image_size': original_size,
            'detections': all_detections
        }
    
    def visualize_detections(
        self,
        detection_results: Dict[str, Any],
        save_path: str = None,
        show_labels: bool = True,
        max_detections: int = 10
    ):
        """
        可视化检测结果
        """
        image_path = detection_results['image_path']
        detections = detection_results['detections']
        
        # 加载原始图像
        image = Image.open(image_path).convert('RGB')
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # 按置信度排序并限制显示数量
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:max_detections]
        
        # 绘制边界框
        colors = plt.cm.Set3(np.linspace(0, 1, len(detections)))
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            # 创建边界框
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor=colors[i],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # 添加标签
            if show_labels:
                label = f"{detection['class_name']}: {detection['confidence']:.2f}"
                ax.text(
                    x, y - 5,
                    label,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7),
                    fontsize=10,
                    color='black'
                )
        
        ax.set_title(f"目标检测结果 - 检测到 {len(detections)} 个物体")
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"检测结果已保存到: {save_path}")
        
        plt.show()
    
    def batch_detect(
        self,
        image_dir: str,
        output_dir: str,
        target_classes: List[int] = None,
        confidence_threshold: float = 0.3
    ):
        """
        批量检测图像
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 获取所有图像文件
        image_files = []
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 批量处理
        results = []
        for i, image_file in enumerate(image_files):
            print(f"\n处理图像 {i+1}/{len(image_files)}: {image_file}")
            
            image_path = os.path.join(image_dir, image_file)
            
            try:
                # 检测
                detection_results = self.detect_objects_in_image(
                    image_path=image_path,
                    target_classes=target_classes,
                    confidence_threshold=confidence_threshold
                )
                
                # 可视化
                output_path = os.path.join(output_dir, f"detection_{image_file}")
                self.visualize_detections(
                    detection_results,
                    save_path=output_path
                )
                
                results.append(detection_results)
                
            except Exception as e:
                print(f"处理图像 {image_file} 时出错: {e}")
        
        # 保存结果
        results_path = os.path.join(output_dir, 'detection_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n批量检测完成，结果保存在: {output_dir}")
        print(f"检测结果JSON文件: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Object Detection Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='./detection_results', help='Output directory')
    parser.add_argument('--target_classes', type=int, nargs='+', help='Target classes to detect')
    parser.add_argument('--confidence_threshold', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples per class')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of ODE steps')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_classes', type=int, default=80, help='Number of classes')
    
    args = parser.parse_args()
    
    # 初始化推理器
    detector = DetectionInference(
        model_path=args.model_path,
        num_classes=args.num_classes,
        device=args.device
    )
    
    if args.image_path:
        # 单张图像检测
        print(f"检测图像: {args.image_path}")
        
        detection_results = detector.detect_objects_in_image(
            image_path=args.image_path,
            target_classes=args.target_classes,
            confidence_threshold=args.confidence_threshold,
            num_samples=args.num_samples,
            num_steps=args.num_steps
        )
        
        # 可视化结果
        detector.visualize_detections(detection_results)
        
        # 打印结果
        print(f"\n检测结果:")
        for detection in detection_results['detections']:
            print(f"  {detection['class_name']}: 置信度 {detection['confidence']:.3f}")
    
    elif args.image_dir:
        # 批量检测
        print(f"批量检测目录: {args.image_dir}")
        
        detector.batch_detect(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            target_classes=args.target_classes,
            confidence_threshold=args.confidence_threshold
        )
    
    else:
        print("请指定 --image_path 或 --image_dir 参数")


if __name__ == "__main__":
    main()


