# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

"""
目标检测Flow Matching演示脚本
展示如何使用训练好的模型进行目标检测
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from detection_flow_matching import ObjectDetectionFlowMatching


def create_demo_image():
    """
    创建一个演示图像，包含简单的几何形状
    """
    # 创建白色背景图像
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # 绘制一些简单的形状作为"物体"
    # 矩形1 (模拟汽车)
    draw.rectangle([50, 100, 150, 180], fill='blue', outline='black', width=2)
    
    # 圆形 (模拟球)
    draw.ellipse([200, 80, 280, 160], fill='red', outline='black', width=2)
    
    # 三角形 (模拟标志)
    draw.polygon([(300, 50), (350, 120), (250, 120)], fill='green', outline='black', width=2)
    
    # 添加一些文字
    draw.text((60, 190), "Car", fill='black')
    draw.text((220, 170), "Ball", fill='black')
    draw.text((280, 130), "Sign", fill='black')
    
    return img


def extract_simple_features(image):
    """
    简单的特征提取（用于演示）
    在实际应用中，这里应该使用预训练的CNN
    """
    # 将图像转换为numpy数组
    img_array = np.array(image)
    
    # 简单的特征：图像的平均颜色和尺寸信息
    features = np.concatenate([
        img_array.mean(axis=(0, 1)),  # RGB平均值
        [img_array.shape[0], img_array.shape[1]],  # 尺寸
        img_array.std(axis=(0, 1))  # RGB标准差
    ])
    
    # 扩展到2048维（模拟ResNet特征）
    features = np.tile(features, 2048 // len(features) + 1)[:2048]
    
    return torch.tensor(features, dtype=torch.float32)


def demo_detection():
    """
    演示目标检测流程
    """
    print("=== Flow Matching目标检测演示 ===\n")
    
    # 1. 创建演示图像
    print("1. 创建演示图像...")
    demo_image = create_demo_image()
    demo_image.save('demo_image.png')
    print("   演示图像已保存为 demo_image.png")
    
    # 2. 初始化检测器
    print("\n2. 初始化检测器...")
    detector = ObjectDetectionFlowMatching(
        num_classes=3,  # 3个类别：汽车、球、标志
        image_feature_dim=2048,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"   使用设备: {detector.device}")
    
    # 3. 提取图像特征
    print("\n3. 提取图像特征...")
    image_features = extract_simple_features(demo_image)
    print(f"   特征维度: {image_features.shape}")
    
    # 4. 对每个类别进行检测
    print("\n4. 开始目标检测...")
    class_names = ['car', 'ball', 'sign']
    
    all_detections = []
    
    for class_id, class_name in enumerate(class_names):
        print(f"\n   检测类别 {class_id}: {class_name}")
        
        # 生成检测结果
        bboxes, confidences = detector.detect_objects(
            image_features=image_features.unsqueeze(0),
            target_class=class_id,
            num_samples=20,  # 生成20个样本
            num_steps=50     # 50步ODE求解
        )
        
        # 后处理
        final_bboxes, final_confidences = detector.post_process_detections(
            bboxes, confidences, 
            confidence_threshold=0.1  # 较低的阈值用于演示
        )
        
        print(f"     生成样本数: {len(bboxes)}")
        print(f"     有效检测数: {len(final_bboxes)}")
        
        if len(final_bboxes) > 0:
            for i, (bbox, conf) in enumerate(zip(final_bboxes, final_confidences)):
                x, y, w, h = bbox.cpu().numpy()
                print(f"       检测 {i+1}: 位置({x:.2f}, {y:.2f}), 尺寸({w:.2f}, {h:.2f}), 置信度{conf:.3f}")
                
                all_detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'bbox': bbox.cpu().numpy(),
                    'confidence': conf.item()
                })
    
    # 5. 可视化结果
    print("\n5. 可视化检测结果...")
    visualize_detections(demo_image, all_detections)
    
    # 6. 总结
    print(f"\n=== 检测完成 ===")
    print(f"总共检测到 {len(all_detections)} 个物体")
    for detection in all_detections:
        print(f"  {detection['class_name']}: 置信度 {detection['confidence']:.3f}")


def visualize_detections(image, detections):
    """
    可视化检测结果
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始图像
    ax1.imshow(image)
    ax1.set_title('原始图像')
    ax1.axis('off')
    
    # 检测结果
    ax2.imshow(image)
    
    if detections:
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            # 绘制边界框
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor=colors[i % len(colors)],
                facecolor='none'
            )
            ax2.add_patch(rect)
            
            # 添加标签
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            ax2.text(
                x, y - 5,
                label,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i % len(colors)], alpha=0.7),
                fontsize=10,
                color='black'
            )
    
    ax2.set_title(f'检测结果 (检测到 {len(detections)} 个物体)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('detection_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   检测结果已保存为 detection_results.png")


def demo_training_process():
    """
    演示训练过程（简化版）
    """
    print("\n=== 训练过程演示 ===")
    
    # 创建一些模拟训练数据
    batch_size = 8
    num_classes = 3
    
    # 模拟图像特征
    image_features = torch.randn(batch_size, 2048)
    
    # 模拟边界框（归一化坐标）
    target_bboxes = torch.rand(batch_size, 4) * 0.5 + 0.25  # 在图像中心区域
    
    # 模拟类别标签
    target_classes = torch.randint(0, num_classes, (batch_size,))
    
    # 初始化检测器
    detector = ObjectDetectionFlowMatching(
        num_classes=num_classes,
        image_feature_dim=2048,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(detector.model.parameters(), lr=1e-4)
    
    print("开始训练演示...")
    
    # 训练几步
    for step in range(5):
        loss_dict = detector.train_step(
            image_features=image_features,
            target_bboxes=target_bboxes,
            target_classes=target_classes,
            optimizer=optimizer
        )
        
        print(f"  步骤 {step+1}: 损失 = {loss_dict['loss']:.4f}")
    
    print("训练演示完成！")


if __name__ == "__main__":
    print("Flow Matching目标检测演示程序")
    print("=" * 50)
    
    # 运行检测演示
    demo_detection()
    
    # 运行训练演示
    demo_training_process()
    
    print("\n演示完成！")
    print("生成的文件:")
    print("  - demo_image.png: 演示图像")
    print("  - detection_results.png: 检测结果可视化")

