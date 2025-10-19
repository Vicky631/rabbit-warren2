# 基于Flow Matching的目标检测

这个项目展示了如何使用Flow Matching技术实现图像中指定物体的目标定位和检测。

## 核心思想

将目标检测问题转化为**条件生成问题**：
- 输入：图像特征 + 目标类别条件
- 输出：边界框坐标 (x, y, w, h) + 置信度分数
- 方法：使用Flow Matching从噪声生成目标物体的位置信息

## 项目结构

```
object_detection/
├── detection_flow_matching.py    # 核心模型实现
├── train_detection.py           # 训练脚本
├── inference_detection.py       # 推理和可视化脚本
└── README.md                    # 说明文档
```

## 主要特性

### 1. 条件生成架构
- **DetectionFlowModel**: 基于Flow Matching的检测模型
- **DetectionFlowWrapper**: 适配Flow Matching框架的包装器
- **ObjectDetectionFlowMatching**: 完整的检测系统

### 2. 训练流程
- 支持COCO格式数据集
- 自动特征提取（ResNet50）
- 端到端训练流程
- 验证和检查点保存

### 3. 推理功能
- 单张图像检测
- 批量图像处理
- 可视化检测结果
- 后处理（NMS、置信度过滤）

## 安装依赖

```bash
# 安装基础依赖
pip install torch torchvision
pip install matplotlib pillow tqdm

# 安装flow_matching库
cd ../../  # 回到项目根目录
pip install -e .
```

## 使用方法

### 1. 训练模型

#### 使用合成数据集（快速开始）
```bash
python train_detection.py \
    --data_dir ./data \
    --batch_size 16 \
    --num_epochs 50 \
    --num_classes 10 \
    --create_synthetic \
    --device cuda
```

#### 使用真实COCO数据集
```bash
# 1. 下载COCO数据集
# 2. 准备标注文件（COCO格式）
# 3. 运行训练
python train_detection.py \
    --data_dir /path/to/coco \
    --batch_size 8 \
    --num_epochs 100 \
    --num_classes 80 \
    --device cuda
```

### 2. 推理检测

#### 单张图像检测
```bash
python inference_detection.py \
    --model_path ./checkpoints/best_model.pth \
    --image_path ./test_image.jpg \
    --target_classes 0 1 2 \
    --confidence_threshold 0.3 \
    --num_samples 50
```

#### 批量检测
```bash
python inference_detection.py \
    --model_path ./checkpoints/best_model.pth \
    --image_dir ./test_images \
    --output_dir ./detection_results \
    --target_classes 0 1 2 \
    --confidence_threshold 0.3
```

## 核心算法原理

### 1. Flow Matching在目标检测中的应用

传统目标检测方法通常使用：
- 分类器预测类别
- 回归器预测边界框

我们的方法使用Flow Matching：
- 将边界框坐标作为生成目标
- 从噪声分布生成目标位置
- 通过条件信息（图像特征+类别）指导生成过程

### 2. 模型架构

```
输入: 图像特征 [batch, 2048] + 类别条件 [batch] + 时间步 [batch]
     ↓
时间嵌入 + 类别嵌入 + 图像特征投影
     ↓
拼接特征 [batch, 4+128+128+512]
     ↓
MLP网络
     ↓
输出: 边界框 [batch, 4] + 置信度 [batch, 1]
```

### 3. 训练过程

1. **数据准备**: 图像特征提取 + 边界框归一化
2. **路径采样**: 从噪声到真实边界框的插值路径
3. **速度场学习**: 训练网络预测速度场
4. **损失计算**: Flow Matching损失函数

### 4. 推理过程

1. **特征提取**: 使用ResNet50提取图像特征
2. **条件生成**: 给定类别条件，从噪声生成边界框
3. **ODE求解**: 使用数值方法求解生成过程
4. **后处理**: NMS + 置信度过滤

## 参数说明

### 训练参数
- `--batch_size`: 批次大小（建议8-32）
- `--num_epochs`: 训练轮数
- `--num_classes`: 类别数量
- `--device`: 计算设备（cuda/cpu）

### 推理参数
- `--target_classes`: 要检测的类别列表
- `--confidence_threshold`: 置信度阈值（0.1-0.9）
- `--num_samples`: 每个类别生成的样本数（10-100）
- `--num_steps`: ODE求解步数（50-200）

## 性能优化建议

### 1. 训练优化
- 使用更大的批次大小（如果GPU内存允许）
- 调整学习率调度策略
- 使用数据增强技术
- 预训练特征提取器

### 2. 推理优化
- 减少ODE求解步数（速度vs精度权衡）
- 使用更少的生成样本
- 并行处理多个类别
- 模型量化加速

### 3. 模型改进
- 使用更强的特征提取器（如EfficientNet、Vision Transformer）
- 添加注意力机制
- 多尺度特征融合
- 改进的后处理算法

## 扩展应用

### 1. 其他检测任务
- **实例分割**: 生成分割掩码
- **关键点检测**: 生成关键点坐标
- **3D目标检测**: 扩展到3D边界框

### 2. 多模态检测
- **文本条件检测**: 根据文本描述检测物体
- **音频条件检测**: 根据声音检测物体
- **多图像融合**: 利用多视角信息

### 3. 实时应用
- **视频目标跟踪**: 结合时序信息
- **移动端部署**: 模型压缩和加速
- **边缘计算**: 轻量化模型设计

## 常见问题

### Q: 为什么选择Flow Matching而不是传统的检测方法？
A: Flow Matching提供了更灵活的生成框架，可以：
- 处理多模态条件信息
- 生成多样化的检测结果
- 更好地处理类别不平衡问题

### Q: 如何提高检测精度？
A: 建议：
- 增加训练数据量
- 使用更强的特征提取器
- 调整超参数（学习率、批次大小等）
- 改进后处理算法

### Q: 模型训练需要多长时间？
A: 取决于：
- 数据集大小
- 模型复杂度
- 硬件配置
- 通常需要几小时到几天

### Q: 如何评估模型性能？
A: 使用标准指标：
- mAP (mean Average Precision)
- IoU (Intersection over Union)
- 检测速度 (FPS)

## 引用

如果您使用了这个项目，请引用原始Flow Matching论文：

```bibtex
@misc{lipman2024flowmatchingguidecode,
      title={Flow Matching Guide and Code}, 
      author={Yaron Lipman and Marton Havasi and Peter Holderrieth and Neta Shaul and Matt Le and Brian Karrer and Ricky T. Q. Chen and David Lopez-Paz and Heli Ben-Hamu and Itai Gat},
      year={2024},
      eprint={2412.06264},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.06264}, 
}
```

## 许可证

本项目遵循CC-BY-NC许可证。详见LICENSE文件。


