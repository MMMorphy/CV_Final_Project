import matplotlib.pyplot as plt
import os
import sys
import numpy as np
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)
from PIL import Image
import utils
import config
import torch
import torchvision

content_loss_0 = []
with open('content_0.txt', 'r') as file:
    for line in file:
        content_loss_0.append(float(line))
content_loss_100 = []
with open('content_100.txt', 'r') as file:
    for line in file:
        content_loss_100.append(float(line))
style_loss_0 = []
with open('style_0.txt', 'r') as file:
    for line in file:
        style_loss_0.append(float(line))
style_loss_100 = []
with open('style_100.txt', 'r') as file:
    for line in file:
        style_loss_100.append(float(line))
laplacian_loss_0 = []
with open('laplacian_0.txt', 'r') as file:
    for line in file:
        laplacian_loss_0.append(float(line))
laplacian_loss_100 = []
with open('laplacian_100.txt', 'r') as file:
    for line in file:
        laplacian_loss_100.append(float(line))

x = np.arange(0, 40000, 500)

content_iamge = torch.Tensor(np.array(Image.open(config.content_image).resize(config.image_size)) / 255.).permute(2, 0, 1).unsqueeze(dim=0)[:, :3, :, :].cuda()
style_image = torch.Tensor(np.array(Image.open(config.style_image).resize(config.image_size)) / 255.).permute(2, 0, 1).unsqueeze(dim=0)[:, :3, :, :].cuda()
lossnet = utils.LossNet(torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)).cuda()
content_feature = lossnet(content_iamge)
style_feature = lossnet(style_image)
style_grams = [utils.gram_matrix(y) for y in style_feature]
content_laplacian = [utils.laplacian(content_iamge, p) for p in config.laplacian_pool_size]

real_time = torch.Tensor(np.array(Image.open('experiments/real-time.png').resize(config.image_size)) / 255.).permute(2, 0, 1).unsqueeze(dim=0)[:, :3, :, :].cuda()
adain = torch.Tensor(np.array(Image.open('experiments/adain.jpg').resize(config.image_size)) / 255.).permute(2, 0, 1).unsqueeze(dim=0)[:, :3, :, :].cuda()
real_time_feature = lossnet(real_time)
real_time_laplacian = [utils.laplacian(real_time, p) for p in config.laplacian_pool_size]
real_time_grams = [utils.gram_matrix(y) for y in real_time_feature]
adain_feature = lossnet(adain)
adain_grams = [utils.gram_matrix(y) for y in adain_feature]
adain_laplacian = [utils.laplacian(adain, p) for p in config.laplacian_pool_size]

content_loss_real_time = torch.nn.functional.mse_loss(real_time_feature[0], content_feature[0])
style_loss_real_time = sum([torch.nn.functional.mse_loss(a, b, reduction='sum') for a, b in zip(style_grams, real_time_grams)])
laplacian_loss_real_time = sum([torch.nn.functional.mse_loss(a, b, reduction='sum') for a, b in zip(content_laplacian, real_time_laplacian)])

content_loss_adain = torch.nn.functional.mse_loss(adain_feature[0], content_feature[0])
style_loss_adain = sum([torch.nn.functional.mse_loss(a, b, reduction='sum') for a, b in zip(style_grams, adain_grams)])
laplacian_loss_adain = sum([torch.nn.functional.mse_loss(a, b, reduction='sum') for a, b in zip(content_laplacian, adain_laplacian)])

plt.plot(x, laplacian_loss_0, label='Gatys-style')
plt.plot(x, laplacian_loss_100, label='Lapstyle')
plt.axhline(laplacian_loss_real_time.cpu(), 0, 40000, label='Real time', color='green')
plt.axhline(laplacian_loss_adain.cpu(), 0, 40000, label='Adain', color='red')
plt.xlabel('iterations')
plt.ylabel(r'$\mathcal{L}_{l}$')
plt.grid()
plt.legend()

plt.show()