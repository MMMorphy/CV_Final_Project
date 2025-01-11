import os
import sys
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models

import torch

import config
import utils

def train(style_image, dataset, visualize):
    dtype = torch.cuda.FloatTensor

    if (visualize):
        img_transform_224 = transforms.Compose([
            transforms.Resize(224),                  # scale shortest side to image_size
            transforms.CenterCrop(224),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
        ])

        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        testImage_pku = utils.load_image("data/content/Peking University.png")
        if testImage_pku.mode == 'RGBA':
            testImage_pku = testImage_pku.convert('RGB')
        testImage_pku = img_transform_224(testImage_pku).repeat(1, 1, 1, 1).type(dtype)
        testImage_pku.requires_grad = False

        testImage_girl = utils.load_image("data/content/val2017/000000006471.jpg")
        if testImage_girl.mode == 'RGBA':
            testImage_girl = testImage_girl.convert('RGB')
        testImage_girl = img_transform_224(testImage_girl).repeat(1, 1, 1, 1).type(dtype)
        testImage_girl.requires_grad = False

        testImage_landscape = utils.load_image("data/content/val2017/000000028285.jpg")
        if testImage_landscape.mode == 'RGBA':
            testImage_landscape = testImage_landscape.convert('RGB')
        testImage_landscape = img_transform_224(testImage_landscape).repeat(1, 1, 1, 1).type(dtype)
        testImage_landscape.requires_grad = False
    
    # 前馈网络、优化器、损失网络、损失函数
    image_transformer = utils.ImageTransformNet().type(dtype)
    optimizer = Adam(image_transformer.parameters(), config.lr) 
    loss_mse = torch.nn.MSELoss(reduction='sum')
    VGG19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    loss_net = utils.LossNet(VGG19).to('cuda:0').eval()

    # 训练数据
    dataset_transform = transforms.Compose([
        transforms.Resize(config.image_size[0]),           # scale shortest side to image_size
        transforms.CenterCrop(config.image_size[0]),      # crop center image_size out
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    train_dataset = utils.CustomImageDataset(dataset, dataset_transform)
    train_loader = DataLoader(train_dataset, batch_size = config.BATCH_SIZE)

    # 该前馈网络对应的风格图像
    style_transform = transforms.Compose([
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    style = utils.load_image(style_image)
    if style.mode == 'RGBA':
            style = style.convert('RGB')
    style = style_transform(style).repeat(config.BATCH_SIZE, 1, 1, 1).type(dtype)
    style.requires_grad = False

    # 计算风格图片的特征以及gram矩阵
    style_features = loss_net(style)
    style_gram = [utils.gram_matrix(fmap) for fmap in style_features]

    for epoch in range(config.EPOCH):
        img_count = 0
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        aggregate_tv_loss = 0.0
        aggregate_laplacian_loss = 0.0

        image_transformer.train()
        for i, x in enumerate(train_loader):
            img_batch_read = len(x)
            img_count += img_batch_read

            optimizer.zero_grad()

            x = x.type(dtype)
            y_hat = image_transformer(x)

            # 计算内容图片的“特征”和laplacian
            y_c_features = loss_net(x)
            y_c_laplacian = [utils.laplacian(x, p) for p in config.laplacian_pool_size]

            y_hat_features = loss_net(y_hat)
            y_hat_gram = [utils.gram_matrix(fmap) for fmap in y_hat_features]
            y_hat_laplacian = [utils.laplacian(y_hat, p) for p in config.laplacian_pool_size]

            # style loss
            style_loss = 0.0
            for a, b in zip(y_hat_gram[1:], style_gram[1:]):
                style_loss += loss_mse(a, b)
            aggregate_style_loss += style_loss.item()

            # content loss
            content_loss = loss_mse(y_c_features[0], y_hat_features[0])
            aggregate_content_loss += content_loss.item()

            # laplacian loss
            laplacian_loss = 0.0
            for gamma, a, b in zip(config.laplacian_weight, y_hat_laplacian, y_c_laplacian):
                laplacian_loss += loss_mse(a, b) * gamma
            aggregate_laplacian_loss += laplacian_loss

            # 我猜这是要求输出图片平滑一些
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = diff_i + diff_j
            aggregate_tv_loss += tv_loss.item()

            # total loss 有四项
            total_loss = style_loss * config.style_weight + content_loss * config.content_weight + tv_loss * config.TV_WEIGHT + laplacian_loss

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            if ((i + 1) % 100 == 0):
                print(f'Epoch {epoch}: [{img_count}/{len(train_dataset)}] --style loss: {aggregate_style_loss / (i+1.0):.8f}\
                      --content loss: {aggregate_content_loss / (i+1.0):.8f}--laplacian loss: {aggregate_laplacian_loss / (i+1.0):.8f}\
                        --tv loss: {aggregate_tv_loss / (i+1.0):.8f}')

            if ((i + 1) % 1000 == 0) and (visualize):
                image_transformer.eval()
                if not os.path.exists("experiments/visualization"):
                    os.makedirs("experiments/visualization")
                if not os.path.exists("experiments/visualization/%s" %config.style_name):
                    os.makedirs("experiments/visualization/%s" %config.style_name)

                outputTestImage_pku = image_transformer(testImage_pku).cpu()
                pku_path = "experiments/visualization/%s/pku_%d_%05d.jpg" %(config.style_name, epoch+1, i+1)
                utils.save_image(pku_path, outputTestImage_pku.data[0])

                outputTestImage_dan = image_transformer(testImage_girl).cpu()
                dan_path = "experiments/visualization/%s/girl_%d_%05d.jpg" %(config.style_name, epoch+1, i+1)
                utils.save_image(dan_path, outputTestImage_dan.data[0])

                outputTestImage_maine = image_transformer(testImage_landscape).cpu()
                maine_path = "experiments/visualization/%s/landscape_%d_%05d.jpg" %(config.style_name, epoch+1, i+1)
                utils.save_image(maine_path, outputTestImage_maine.data[0])

                print("images saved")
                image_transformer.train()

    image_transformer.eval()
    image_transformer.cpu()
    filename = "models/" + config.style_name + ".model"
    torch.save(image_transformer.state_dict(), filename)

train(config.style_image, config.dataset, visualize=True)    