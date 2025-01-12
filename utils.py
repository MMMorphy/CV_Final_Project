import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset
import os

def gram_matrix(x):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

class LossNet(nn.Module):
    def __init__(self, backbone): # 默认我们要用VGG19
        super(LossNet, self).__init__()
        # 选择内容层和风格层的索引
        # 内容层：conv4_2 --- 12层
        # 风格层：conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 --- 0, 3, 6, 11, 16层
        # 内容层放在输出的第0个，风格层使用索引[1:]即可
        self.select = ['12', '0', '3', '6', '11', '16']
        self.feature_detector = backbone.features
        # 冻结参数
        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        features = []
        for name, layer in self.feature_detector._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

def laplacian(x, p: int):
    # 注意！！这个函数包括了池化
    x_down = F.avg_pool2d(x, kernel_size=p, stride=p)

    b, ch, h, w = x_down.size()
    lap_kernel = torch.tensor([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]], dtype=torch.float32).cuda()
    lap_kernel = lap_kernel.unsqueeze(0).unsqueeze(0)
    lap_kernel = lap_kernel.expand(ch, ch, -1, -1)
    res = F.conv2d(x_down, lap_kernel, padding=1)
    res = torch.sum(res, dim=1)
    
    return res

# opens and returns image file as a PIL image (0-255)
def load_image(filename):
    img = Image.open(filename).convert('RGB')
    return img

# assumes data comes in batch form (ch, h, w)
def save_image(filename, data):
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.clone().numpy()
    img = ((img * std + mean).transpose(1, 2, 0)*255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)
    
# using ImageNet values
def normalize_tensor_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out
    
class ImageTransformNet(nn.Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()
        self.padding = nn.ReflectionPad2d(40)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu4 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu5 = nn.ReLU()
        self.deconv3 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.padding(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.residual_blocks(x)
        x = self.relu4(self.deconv1(x))
        x = self.relu5(self.deconv2(x))
        x = self.tanh(self.deconv3(x))
        return x
    
class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # 获取所有图像文件的路径
        self.image_paths = [
            os.path.join(root, fname) 
            for fname in os.listdir(root) 
            if fname.endswith(('jpg', 'png'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # 确保图像是 RGB 格式
        if self.transform:
            image = self.transform(image)
        return image