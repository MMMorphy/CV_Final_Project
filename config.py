style_weight = 1000
content_weight = 1
laplacian_weight = [100, 100]
laplacian_pool_size = [4, 16]

content_image = './data/content/Peking University.png'
style_image = './data/style/The starry night.png'
image_size = (224, 224)

optimizer = 'adam' # 还可以填 lbfgs
lr = 0.001