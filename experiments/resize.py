import os
import sys
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)
from PIL import Image

def resize_image(input_path, output_path, new_width, new_height):
    with Image.open(input_path) as img:
        resized_img = img.resize((new_width, new_height))
        resized_img.save(output_path)
        print(f"Image resized and saved to {output_path}")

resize_image(r"C:\Users\HP1\Desktop\files\course file\2.1\computer vision\homework\Final Project\CV_Final_Project\data\content\val2017\000000020059.jpg", 'change.png', 256, 256)