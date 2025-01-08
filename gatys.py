import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import torch.optim as optim

# Load images
img0 = Image.open('./style.png').resize([224, 224])
style_img = torch.Tensor(np.array(img0) / 255.)
style_img = style_img.permute(2, 0, 1).unsqueeze(dim=0)
style_img = style_img[:, :3, :, :]

img1 = Image.open('./pku.png').resize([224, 224])
content_img = torch.Tensor(np.array(img1) / 255.)
content_img = content_img.permute(2, 0, 1).unsqueeze(dim=0)
content_img = content_img[:, :3, :, :]

# Define the LossNet class
class LossNet(torch.nn.Module):
    def __init__(self, backbone):
        super(LossNet, self).__init__()
        self.select = ['3', '8', '15', '22']  # Layers to extract features from
        self.feature_detector = backbone.features
        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        features = []
        for name, layer in self.feature_detector._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

# Load VGG16 model
vgg16 = torchvision.models.vgg16(pretrained=True)
lossnet = LossNet(vgg16).to('cuda:0').eval()

# Extract features from content image
content_features = lossnet(content_img.cuda())
print(len(content_features))
print(content_features[0].shape)

# Function to compute Gram matrix
def gram_matrix(x):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

# Extract features from style image and compute Gram matrix
style_features = lossnet(style_img.cuda())
style_grams = [gram_matrix(x) for x in style_features]
print(len(style_features))
print(style_features[0].shape)
print(len(style_grams))
print(style_grams[0].shape)

# Style transfer training function
def style_transfer_train(model, input_img, style_grams, content_features, optimizer, max_T=1000):
    style_weight = 1e7
    content_weight = 1

    for t in range(max_T):
        optimizer.zero_grad()
        features = model(input_img)
        grams = [gram_matrix(x) for x in features]

        content_loss = F.mse_loss(features[2], content_features[2]) * content_weight
        style_loss = 0
        for a, b in zip(grams, style_grams):
            style_loss += F.mse_loss(a, b) * style_weight

        loss = style_loss + content_loss
        loss.backward()
        optimizer.step()

        if (t + 1) % 500 == 0:
            print(f'Step {t + 1}: Total Loss: {loss.item():.8f} - Style Loss: {style_loss.item():.8f} - Content Loss: {content_loss.item():.8f}')

# Set up optimizer
input_img = content_img.clone().cuda()
input_img.requires_grad = True
optimizer = optim.Adam([input_img], lr=0.001)

# Train the model
style_transfer_train(lossnet, input_img, style_grams, content_features, optimizer, max_T=10000)

# Convert result to a displayable format
result = input_img.data.squeeze(dim=0).permute(1, 2, 0)
result = result.cpu().numpy()  # 转换为 NumPy 数组
result = np.clip(result, 0, 1)  # 限制在 [0, 1] 之间

# Display the images
fig, ax = plt.subplots(1, 3)
fig.set_figheight(10)
fig.set_figwidth(30)
ax[0].imshow(np.array(img0) / 255.)
ax[0].set_title('Content Image', fontsize=15)
ax[1].imshow(np.array(img1) / 255.)
ax[1].set_title('Style Image', fontsize=15)
ax[2].imshow(result)
ax[2].set_title('Transferred Image', fontsize=15)
plt.show()
plt.savefig('output_image.png')
