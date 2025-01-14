import argparse
import torch
from torchvision import transforms
from PIL import Image
import os
import sys
from utils import ImageTransformNet

def load_model(style: str) -> ImageTransformNet:
    model_paths = {
        "star": "models/star.model",
        "ink": "models/ink.model",
        "mosaic": "models/mosaic.model"
    }
    
    if style not in model_paths:
        raise ValueError(f"Invalid style choice: {style}. Choose from {list(model_paths.keys())}.")
    
    model = ImageTransformNet()
    model.load_state_dict(torch.load(model_paths[style], weights_only=True))
    model.eval()
    return model

def preprocess_image(image_path: str, args):
    image = Image.open(image_path).convert('RGB')
    if args.style == 'ink':
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.style == 'star' or 'mosaic':
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError('Only ink, star and mosaic allowed')
    return transform(image).unsqueeze(0)

def postprocess_and_save(output_tensor, output_path: str, args):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    if args.style == 'ink':
        transform = transforms.Compose([
            transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444]),
            transforms.ToPILImage()
        ])
    elif args.style == 'star' or 'mosaic':
        transform = transforms.Compose([
            transforms.ToPILImage()
        ])
    else:
        raise ValueError('Only ink, star and mosaic allowed')

    image = transform(output_tensor.squeeze(0).detach().cpu())
    image.save(output_path)
    print(f"Stylized image saved at: {output_path}")

def process_directory(content_dir: str, output_dir: str, model, args):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(content_dir):
        if filename.lower().endswith(('.jpg', '.png')):
            content_path = os.path.join(content_dir, filename)
            output_path = os.path.join(output_dir, f"stylized_{filename}")
            
            content_image = preprocess_image(content_path, args)
            
            with torch.no_grad():
                output_image = model(content_image)
            
            postprocess_and_save(output_image, output_path, args)

def main(args):
    model = load_model(args.style)
    
    if os.path.isdir(args.content):
        process_directory(args.content, args.output, model, args)
    else:
        raise ValueError("The content path must be a directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-time Transfer")
    parser.add_argument('--content', type=str, required=True, help="Path to the content image directory")
    parser.add_argument('--style', type=str, required=True, help="Style: star, ink, mosaic")
    parser.add_argument('--output', type=str, default='output', help="Path to the output directory")

    args = parser.parse_args()
    main(args)
