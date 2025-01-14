style_weight = 1e4   #####
content_weight = 1
laplacian_weight = [100, 100]   #####
laplacian_pool_size = [4, 16]
TV_WEIGHT = 1e-7

dataset = 'data/content/val2017'
save = 1    ######
image_size = (256, 256)

prototxt_path = './models/VGG_ILSVRC_19_layers_deploy.prototxt'
weight_path = './models/VGG_ILSVRC_19_layers.caffemodel'

optimizer = 'adam'
lr = 0.00012   ######
max_T = 40000  ######
BATCH_SIZE = 4
EPOCH = 10
visualize = True

# You can adjust the following parameters as a user
content_image = "data/content/Peking University.png"    ######
style_image = './data/style/The starry night.png'   ######
output_path = './experiments/gatys_star_1e4_100_4w_pku.png'  ######