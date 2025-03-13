# CV_Final_Project
2024fall cv final project

# Environment Setup and Running Guide

## 1. Install Python and Dependencies

### 1.1 Create a Conda Environment

First, make sure you have installed [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual), then follow these steps to create a new Conda environment:

```bash
conda create --name style_transfer python=3.11
conda activate style_transfer
```
### 1.2Install PyTorch-2.5.1 using Conda
Install PyTorch-2.5.1 according to the official documentation:
```bash
conda install pytorch==2.5.1 torchvision torchaudio -c pytorch
```

### 1.3 Install Necessary Python Libraries
Run the following commands to install the required Python packages in the activated Conda environment:
```bash
conda install numpy 
conda install torch torchvision 
conda install matplotlib  
conda install pillow      
conda install configargparse  
conda install time 
```
## 2  Configure the Project
### 2.1 Dataset and Style Image

Use the `--content` and `--style` flags to provide the respective paths for the content and style images. Place your content images in the data/content/folder, and style images in the data/style/folder.
```bash
mkdir -p data/content
mkdir -p data/style
```

### 2.2 Running Files
You can run the following three files: `gatys.py`, `lapstyle.py`, and `real-time.py`.

#### 2.2.1 `gatys.py` and `lapstyle.py`

 Use `--content_image` and `--style_image` to provide the respective path to the content and style image Use `--output_image` and `--max_T`(default=30000) to set the path to the output and max number of iteration, For example：
 ```python
python gatys.py --content_image data/content/duck.jpg --style_image data/style/mnls.jpg --output_image test.png --max_T 40000
python lapstyle.py --content_image data/content/duck.jpg --style_image data/style/mnls.jpg --output_image test.png --max_T 40000
```

#### 2.2.2 `real-time.py`
Use `--content` and `--style` to provide the respective path to the content and style image. Use `--output` (default: the position of real-time.py) to set the path to the output, For example：
```python
python real-time.py --content data/content --style ink --output output
```
You can also use `--content` to set the path to a content image for a single image style transfer, for example:
```python
python real-time.py --content data/content/duck.jpg --style ink
```


### 2.3 Output Results
After training finishes, the stylized image generated will be saved in the output_path specified in the configuration file. For example:
```bash
output/stylized_image.png
```
