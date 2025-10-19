# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

"""
基于Flow Matching的目标检测实现
将目标检测问题转化为条件生成问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, Any

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import FlowMatchingLoss


class DetectionFlowModel(nn.Module):
    """
    基于Flow Matching的目标检测模型
    输入：图像特征 + 时间步 + 目标类别条件
    输出：边界框坐标和置信度
    """
    
    def __init__(
        self,
        image_feature_dim: int = 2048,  # 图像特征维度
        num_classes: int = 80,          # 类别数量
        bbox_dim: int = 4,              # 边界框维度 (x, y, w, h)
        hidden_dim: int = 512,
        time_embed_dim: int = 128,
        class_embed_dim: int = 128
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.bbox_dim = bbox_dim
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 类别嵌入
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        
        # 图像特征投影
        self.image_proj = nn.Linear(image_feature_dim, hidden_dim)
        
        # 主网络
        self.net = nn.Sequential(
            nn.Linear(bbox_dim + time_embed_dim + class_embed_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, bbox_dim + 1)  # bbox + confidence
        )
        
    def forward(
        self, 
        bbox: Tensor,           # 当前边界框 [batch_size, 4]
        t: Tensor,              # 时间步 [batch_size]
        image_features: Tensor, # 图像特征 [batch_size, image_feature_dim]
        class_condition: Tensor # 类别条件 [batch_size]
    ) -> Tensor:
        """
        前向传播
        """
        batch_size = bbox.shape[0]
        
        # 时间嵌入
        t_emb = self.time_embed(t.unsqueeze(-1))  # [batch_size, time_embed_dim]
        
        # 类别嵌入
        class_emb = self.class_embed(class_condition)  # [batch_size, class_embed_dim]
        
        # 图像特征投影
        img_emb = self.image_proj(image_features)  # [batch_size, hidden_dim]
        
        # 拼接所有特征
        x = torch.cat([bbox, t_emb, class_emb, img_emb], dim=-1)
        
        # 通过主网络
        output = self.net(x)  # [batch_size, 5] (4 for bbox + 1 for confidence)
        
        return output


class DetectionFlowWrapper(ModelWrapper):
    """
    检测模型的包装器，适配Flow Matching框架
    """
    
    def __init__(self, model: DetectionFlowModel):
        super().__init__(model)
        
    def forward(
        self, 
        x: Tensor, 
        t: Tensor, 
        image_features: Tensor,
        class_condition: Tensor,
        **extras
    ) -> Tensor:
        """
        适配Flow Matching的forward接口
        """
        return self.model(
            bbox=x,
            t=t,
            image_features=image_features,
            class_condition=class_condition
        )


class ObjectDetectionFlowMatching:
    """
    基于Flow Matching的目标检测主类
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        image_feature_dim: int = 2048,
        device: str = "cuda"
    ):
        self.device = device
        self.num_classes = num_classes
        
        # 初始化模型
        self.model = DetectionFlowModel(
            image_feature_dim=image_feature_dim,
            num_classes=num_classes
        ).to(device)
        
        # 包装模型
        self.wrapped_model = DetectionFlowWrapper(self.model)
        
        # 初始化Flow Matching组件
        self.scheduler = CondOTScheduler()
        self.path = AffineProbPath(scheduler=self.scheduler)
        self.solver = ODESolver(model=self.wrapped_model, path=self.path)
        
        # 损失函数
        self.loss_fn = FlowMatchingLoss(path=self.path)
        
    def train_step(
        self,
        image_features: Tensor,
        target_bboxes: Tensor,
        target_classes: Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        训练一步
        """
        batch_size = target_bboxes.shape[0]
        
        # 生成随机噪声作为起始点
        x_0 = torch.randn_like(target_bboxes)  # 噪声边界框
        
        # 目标边界框
        x_1 = target_bboxes
        
        # 随机时间步
        t = torch.rand(batch_size, device=self.device)
        
        # 采样路径
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        
        # 模型预测
        model_output = self.wrapped_model(
            x=path_sample.x_t,
            t=t,
            image_features=image_features,
            class_condition=target_classes
        )
        
        # 计算损失
        loss = self.loss_fn(path_sample=path_sample, model_output=model_output)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {"loss": loss.item()}
    
    def detect_objects(
        self,
        image_features: Tensor,
        target_class: int,
        num_samples: int = 10,
        num_steps: int = 50
    ) -> Tuple[Tensor, Tensor]:
        """
        检测指定类别的物体
        
        Args:
            image_features: 图像特征 [1, feature_dim]
            target_class: 目标类别
            num_samples: 生成样本数量
            num_steps: ODE求解步数
            
        Returns:
            bboxes: 检测到的边界框 [num_samples, 4]
            confidences: 置信度分数 [num_samples, 1]
        """
        self.model.eval()
        
        with torch.no_grad():
            # 准备输入
            batch_size = num_samples
            image_features = image_features.expand(batch_size, -1)
            class_condition = torch.full((batch_size,), target_class, device=self.device)
            
            # 从噪声开始采样
            x_init = torch.randn(batch_size, 4, device=self.device)
            
            # 使用ODE求解器生成
            solution = self.solver.sample(
                x_init=x_init,
                step_size=1.0 / num_steps,
                verbose=False
            )
            
            # 获取最终结果
            final_bboxes = solution.x[-1]  # [batch_size, 4]
            
            # 计算置信度
            t_final = torch.ones(batch_size, device=self.device)
            model_output = self.wrapped_model(
                x=final_bboxes,
                t=t_final,
                image_features=image_features,
                class_condition=class_condition
            )
            
            confidences = torch.sigmoid(model_output[:, -1:])  # 置信度
            
            return final_bboxes, confidences
    
    def post_process_detections(
        self,
        bboxes: Tensor,
        confidences: Tensor,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5
    ) -> Tuple[Tensor, Tensor]:
        """
        后处理检测结果：过滤低置信度检测和NMS
        """
        # 过滤低置信度
        valid_mask = confidences.squeeze() > confidence_threshold
        if not valid_mask.any():
            return torch.empty(0, 4, device=self.device), torch.empty(0, 1, device=self.device)
        
        valid_bboxes = bboxes[valid_mask]
        valid_confidences = confidences[valid_mask]
        
        # 简化的NMS (实际应用中应使用更复杂的实现)
        # 这里提供一个基础版本
        if len(valid_bboxes) <= 1:
            return valid_bboxes, valid_confidences
        
        # 按置信度排序
        sorted_indices = torch.argsort(valid_confidences.squeeze(), descending=True)
        keep_indices = []
        
        for i in sorted_indices:
            if i in keep_indices:
                continue
                
            keep_indices.append(i)
            
            # 计算IoU并移除重叠的检测
            current_bbox = valid_bboxes[i:i+1]
            remaining_bboxes = valid_bboxes[sorted_indices[sorted_indices > i]]
            
            if len(remaining_bboxes) > 0:
                ious = self._compute_iou(current_bbox, remaining_bboxes)
                to_remove = torch.where(ious > nms_threshold)[0]
                
                for idx in to_remove:
                    original_idx = sorted_indices[sorted_indices > i][idx]
                    if original_idx not in keep_indices:
                        keep_indices.append(original_idx)
        
        keep_indices = torch.tensor(keep_indices, device=self.device)
        return valid_bboxes[keep_indices], valid_confidences[keep_indices]
    
    def _compute_iou(self, bbox1: Tensor, bbox2: Tensor) -> Tensor:
        """
        计算边界框的IoU
        """
        # 简化的IoU计算
        x1_min, y1_min, x1_max, y1_max = bbox1[0, 0], bbox1[0, 1], bbox1[0, 0] + bbox1[0, 2], bbox1[0, 1] + bbox1[0, 3]
        
        x2_min = bbox2[:, 0]
        y2_min = bbox2[:, 1] 
        x2_max = bbox2[:, 0] + bbox2[:, 2]
        y2_max = bbox2[:, 1] + bbox2[:, 3]
        
        # 计算交集
        inter_x_min = torch.max(x1_min, x2_min)
        inter_y_min = torch.max(y1_min, y2_min)
        inter_x_max = torch.min(x1_max, x2_max)
        inter_y_max = torch.min(y1_max, y2_max)
        
        inter_area = torch.clamp(inter_x_max - inter_x_min, min=0) * torch.clamp(inter_y_max - inter_y_min, min=0)
        
        # 计算并集
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-6)


# 使用示例
def example_usage():
    """
    使用示例
    """
    # 初始化检测器
    detector = ObjectDetectionFlowMatching(
        num_classes=80,  # COCO数据集类别数
        image_feature_dim=2048,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 模拟训练数据
    batch_size = 32
    image_features = torch.randn(batch_size, 2048, device=detector.device)
    target_bboxes = torch.rand(batch_size, 4, device=detector.device)  # 归一化的边界框
    target_classes = torch.randint(0, 80, (batch_size,), device=detector.device)
    
    # 训练
    optimizer = torch.optim.Adam(detector.model.parameters(), lr=1e-4)
    
    for epoch in range(100):
        loss_dict = detector.train_step(image_features, target_bboxes, target_classes, optimizer)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss_dict['loss']:.4f}")
    
    # 检测
    test_image_features = torch.randn(1, 2048, device=detector.device)
    target_class = 0  # 检测类别0的物体
    
    bboxes, confidences = detector.detect_objects(
        image_features=test_image_features,
        target_class=target_class,
        num_samples=20
    )
    
    # 后处理
    final_bboxes, final_confidences = detector.post_process_detections(
        bboxes, confidences, confidence_threshold=0.3
    )
    
    print(f"检测到 {len(final_bboxes)} 个物体")
    print(f"边界框: {final_bboxes}")
    print(f"置信度: {final_confidences}")


if __name__ == "__main__":
    example_usage()


