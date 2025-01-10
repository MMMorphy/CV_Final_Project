import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

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
        # 内容层：conv4_2 --- 21层
        # 风格层：conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 --- 0, 5, 10, 19, 28层
        # 内容层放在输出的第0个，风格层使用索引[1:]即可
        self.select = ['21', '0', '5', '10', '19', '28']
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
    img = Image.open(filename)
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
