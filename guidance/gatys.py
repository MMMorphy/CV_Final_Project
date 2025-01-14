import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
import torchvision
from torchvision.models import VGG19_Weights
import matplotlib.pyplot as plt
import utils

def main(args):
    # 设置运行目录为脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)
    
    # 加载模型
    VGG19 = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT)
    lossnet = utils.LossNet(VGG19).to('cuda:0').eval()

    # 加载内容图像和风格图像
    img0 = Image.open(args.style_image).resize((256, 256))
    style_img = torch.Tensor(np.array(img0) / 255.).cuda().permute(2, 0, 1).unsqueeze(dim=0)[:, :3, :, :]

    img1 = Image.open(args.content_image).resize((256, 256))
    content_img = torch.Tensor(np.array(img1) / 255.).cuda().permute(2, 0, 1).unsqueeze(dim=0)[:, :3, :, :]

    # 计算特征和拉普拉斯变换
    content_features = lossnet(content_img)
    style_features = lossnet(style_img)
    style_grams = [utils.gram_matrix(x) for x in style_features]
    content_laplacian = [utils.laplacian(content_img, p) for p in [4, 16]]

    input_img = content_img.clone().cuda()
    input_img.requires_grad = True

    optimizer = optim.Adam([input_img], lr=0.00012)

    # 优化过程
    for t in range(args.max_steps):
        optimizer.zero_grad()
        features = lossnet(input_img)
        grams = [utils.gram_matrix(x) for x in features]
        content_loss = F.mse_loss(features[0], content_features[0])
        style_loss = sum([F.mse_loss(a, b, reduction='sum') for a, b in zip(grams[1:], style_grams[1:])])
        laplacian_loss = sum([
            F.mse_loss(cl, utils.laplacian(input_img, p), reduction='sum')
            for cl, p in zip(content_laplacian, [4, 16])
        ])
        loss = content_loss * 1.0 + style_loss * 1e4
        loss.backward()
        optimizer.step()

        if t % 500 == 499:
            print(f'Step {t + 1}: Total Loss: {loss.item():.8f} - Style Loss: {style_loss.item():.8f} - Content Loss: {content_loss.item():.8f} - Laplacian Loss: {laplacian_loss.item():.8f}')

    # 保存结果
    result = input_img.data.squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
    result = np.clip(result, 0, 1)

    plt.imsave(args.output_image, result)
    print(f"Stylized image saved to {args.output_image}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LapStyle Image Transfer")
    parser.add_argument('--content_image', type=str, required=True, help="Path to the content image")
    parser.add_argument('--style_image', type=str, required=True, help="Path to the style image")
    parser.add_argument('--output_image', type=str, required=True, help="Path to save the output stylized image")
    parser.add_argument('--max_steps', type=int, default=30000, help="Maximum number of optimization steps (default: 30000)")
    args = parser.parse_args()

    main(args)
