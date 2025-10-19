import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
import torchvision.models as models
import matplotlib.pyplot as plt


# 目标图像加载函数
def load_image(image_path: str, size: tuple = (256, 256)):
    image = Image.open(image_path).convert('RGB')  # 加载并转为RGB模式
    transform = transforms.Compose([
        transforms.Resize(size),  # 调整图像尺寸
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
    ])
    return transform(image).unsqueeze(0)  # 增加batch维度


# 提取图像特征的卷积神经网络 (例如 ResNet)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # 使用预训练ResNet50
        self.resnet.fc = nn.Identity()  # 去掉全连接层，保留特征提取部分

    def forward(self, x):
        return self.resnet(x)


# 定义生成器模型（基于MLP）
class SimpleGenerator(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=256 * 256 * 3):
        super(SimpleGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # 输出值从[-1, 1]映射到图像像素范围
        )

    def forward(self, x):
        return self.fc(x).view(-1, 3, 256, 256)  # 生成一个 256x256 的图像


# 定义训练的生成模型
def train_model(image_path):
    # 加载目标图像
    target_image = load_image(image_path)
    feature_extractor = FeatureExtractor().to('cuda')
    target_features = feature_extractor(target_image.to('cuda'))

    # 使用噪声和目标图像特征生成新图像
    noise = torch.randn(1, 128).to('cuda')  # 随机噪声
    generator = SimpleGenerator(input_dim=128 + target_features.shape[1]).to('cuda')  # 生成器
    input_data = torch.cat([noise, target_features], dim=1)  # 将噪声和图像特征拼接在一起

    generated_image = generator(input_data)

    return generated_image


# 使用 flow_matching 进行训练和生成
def flow_matching_generation(image_path):
    # flow_matching初始化
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)
    loss_fn = MixturePathGeneralizedKL(path=path)

    # 假设这里使用训练数据作为输入 (例如，目标图像的特征)
    # 使用流匹配进行目标图像的生成
    n_samples = 100  # 假设生成100个样本
    x_init = torch.randint(size=(n_samples, 2), high=128, device='cuda')  # 假设生成2D离散数据
    solver = MixtureDiscreteEulerSolver(model=None, path=path, vocabulary_size=128)  # 模型为空，流路径是重点

    # 使用solver进行采样
    sol = solver.sample(x_init=x_init, step_size=1 / 64, verbose=True, return_intermediates=True,
                        time_grid=torch.linspace(0, 1, 10))

    return sol


# 可视化生成的图像
def visualize_generated_image(generated_image):
    generated_image = generated_image.cpu().detach().numpy().transpose(0, 2, 3, 1)[0]
    generated_image = (generated_image + 1) / 2  # 将值从[-1,1]映射到[0,1]
    plt.imshow(generated_image)
    plt.axis('off')
    plt.show()


# 主函数
if __name__ == "__main__":
    image_path = 'your_image_path.jpg'  # 替换为目标图像路径

    # 训练生成模型
    generated_image = train_model(image_path)
    visualize_generated_image(generated_image)

    # 使用flow_matching进行图像生成
    sol = flow_matching_generation(image_path)
    print(sol)  # 显示生成的流路径
