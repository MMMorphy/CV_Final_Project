import os
import sys
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
import torchvision
from torchvision.models import VGG19_Weights
import matplotlib.pyplot as plt
import config
import utils

def lapstyle_transfer_train(lossnet, input_image, style_grams, content_features, content_laplacian, optimizer, max_T=1000):
    for t in range(max_T):
        optimizer.zero_grad()
        features = lossnet(input_image)
        grams = [utils.gram_matrix(x) for x in features]
        # 接下来分别计算损失误差
        content_loss = F.mse_loss(features[0], content_features[0]) # 这是内容误差
        style_loss = 0
        for a, b in zip(grams[1:], style_grams[1:]):
            style_loss += F.mse_loss(a, b, reduction='sum') # 这是风格损失
        laplacian_loss = 0
        for gamma, p, cl in zip(config.laplacian_weight, config.laplacian_pool_size, content_laplacian):
            laplacian_loss += F.mse_loss(cl, utils.laplacian(input_image, p), reduction='sum') * gamma

        loss = content_loss * config.content_weight + style_loss * config.style_weight + laplacian_loss
        loss.backward()
        optimizer.step()

        if (t + 1) % 500 == 0:
            print(f'Step {t + 1}: Total Loss: {loss.item():.8f} - Style Loss: {style_loss.item():.8f} - Content Loss: {content_loss.item():.8f} - Laplacian Loss: {laplacian_loss.item():.8f}')

VGG19 = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT)
lossnet = utils.LossNet(VGG19).to('cuda:0').eval()

img0 = Image.open(config.style_image).resize(config.image_size)
style_img = torch.Tensor(np.array(img0) / 255.).cuda()
style_img = style_img.permute(2, 0, 1).unsqueeze(dim=0)
style_img = style_img[:, :3, :, :]

img1 = Image.open(config.content_image)
content_img = torch.Tensor(np.array(img1.resize(config.image_size)) / 255.).cuda()
content_img = content_img.permute(2, 0, 1).unsqueeze(dim=0)
content_img = content_img[:, :3, :, :]

content_features = lossnet(content_img)
style_features = lossnet(style_img)
style_grams = [utils.gram_matrix(x) for x in style_features]
content_laplacian = [utils.laplacian(content_img, p) for p in config.laplacian_pool_size]

#input_img = torch.rand_like(content_img, dtype=torch.float32).cuda()
#input_img = style_img.clone().cuda()
input_img = content_img.clone().cuda()
input_img.requires_grad = True


# 定义闭包函数
def closure():
    optimizer.zero_grad()  # 清空梯度
    input_img_cont = input_img.contiguous()
    features = lossnet(input_img_cont)  # 计算特征
    grams = [utils.gram_matrix(x) for x in features]

    # 计算内容损失
    content_loss = F.mse_loss(features[0], content_features[0])

    # 计算风格损失
    style_loss = sum(F.mse_loss(a, b) for a, b in zip(grams[1:], style_grams[1:]))

    # 计算拉普拉斯损失
    laplacian_loss = sum(
        F.mse_loss(cl, utils.laplacian(input_img_cont, p)) * gamma
        for gamma, p, cl in zip(config.laplacian_weight, config.laplacian_pool_size, content_laplacian)
    )

    # 总损失
    loss = config.content_weight * content_loss + config.style_weight * style_loss + laplacian_loss
    loss.backward()  # 反向传播计算梯度

    # 打印损失信息
    if closure.counter % 500 == 0:
        print(f'Step {closure.counter}: Total Loss: {loss.item():.8f} - Style Loss: {style_loss.item():.8f} - Content Loss: {content_loss.item():.8f} - Laplacian Loss: {laplacian_loss.item():.8f}')
    
    closure.counter += 1
    return loss

if config.optimizer == 'adam':
    optimizer = optim.Adam([input_img], lr=config.lr)
    lapstyle_transfer_train(lossnet, input_img, style_grams, content_features, content_laplacian, optimizer, config.max_T)
elif config.optimizer == 'lbfgs':
    optimizer = optim.LBFGS([input_img], lr=config.lr)
    closure.counter = 0  # 初始化计数器
    max_iterations = 1000
    for step in range(max_iterations):
        input_img = input_img.contiguous().float()
        optimizer.step(closure)
else:
    raise ValueError('Just Adam or L-BFGS')

result = input_img.data.squeeze(dim=0).permute(1, 2, 0)
result = result.cpu().numpy()  # 转换为 NumPy 数组
result = np.clip(result, 0, 1)  # 限制在 [0, 1] 之间

fig, ax = plt.subplots(1, 3)
fig.set_figheight(10)
fig.set_figwidth(30)
ax[0].imshow(np.array(img0) / 255.)
ax[0].set_title('Style Image', fontsize=15)
ax[1].imshow(np.array(img1) / 255.)
ax[1].set_title('Content Image', fontsize=15)
ax[2].imshow(result)
ax[2].set_title('Stylized Image', fontsize=15)
plt.show()
if config.save:
    plt.imsave(config.output_path, result.resize(img1.size()))