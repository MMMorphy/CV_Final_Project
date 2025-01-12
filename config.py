style_weight = 1e4
content_weight = 1
laplacian_weight = [100, 100]
laplacian_pool_size = [4, 16]
TV_WEIGHT = 1e-7

content_image = "data/content/val2017/000000020059.jpg"
style_image = './data/style/The starry night.png'
style_name = 'starry_night'
output_path = './experiments/lapstyle_vangogh_1e4_100_2w_zebra.png'
dataset = 'data/content/val2017'
save = 1
image_size = (256, 256)

prototxt_path = './models/VGG_ILSVRC_19_layers_deploy.prototxt'
weight_path = './models/VGG_ILSVRC_19_layers.caffemodel'

optimizer = 'adam' # 还可以填 lbfgs
lr = 0.0001
max_T = 20000
BATCH_SIZE = 4
EPOCH = 10
visualize = True