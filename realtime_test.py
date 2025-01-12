import torch
from torchvision import transforms
from PIL import Image
import utils  # 假设 utils 包含 `ImageTransformNet` 和 `save_image` 方法
import config  # 假设配置文件仍然存在
import time 
import os
def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = utils.ImageTransformNet()  # 定义模型结构
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    return model
def preprocess_image(image_path, image_size=512, device='cuda' if torch.cuda.is_available() else 'cpu'):
    img = utils.load_image(image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # utils.normalize_tensor_transform()  # 如果训练中使用了归一化，这里需要一致
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # 添加 batch 维度
    return img_tensor
def save_output_image(tensor, output_path):
    output_tensor = tensor.cpu().squeeze(0)  # 移除 batch 维度
    output_image = transforms.ToPILImage()(output_tensor)
    output_image.save(output_path)
def batch_style_transfer(input_folder, output_folder, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, device)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    image_files.sort()  # 按文件名排序

    total_start_time = time.time()  # 开始总计时

    for idx, image_file in enumerate(image_files[:10], start=1):  # 只处理前 10 张图片
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"{idx:03d}_fango_gatys_test.jpg")  # 按序号命名输出图片
        
        start_time = time.time()
        input_tensor = preprocess_image(input_path, device=device)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        save_output_image(output_tensor, output_path)
        end_time = time.time()
        
        print(f"Processed {image_file} -> {output_path} in {end_time - start_time:.2f} seconds")

    total_end_time = time.time()  # 结束总计时
    print(f"Processed 10 images in total time: {total_end_time - total_start_time:.2f} seconds")

input_image = "Final_Project/cv_final_test"  # 测试图片路径
output_image = "Final_Project/cv_final_test"  # 保存结果路径
trained_model = "/home/stu2200011612/Final_Project/models/starry_night.model"  # 训练好的模型路径

batch_style_transfer(input_image, output_image, trained_model)