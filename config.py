style_weight = 1e3
content_weight = 1
laplacian_weight = [10, 10]
laplacian_pool_size = [4, 16]

content_image = './data/content/Peking University.png'
style_image = './data/style/The starry night.png'
output_path = './experiments/lapstyle_pku_vangogh_lbfgs.png'
save = False
image_size = (224, 224)

prototxt_path = './models/VGG_ILSVRC_19_layers_deploy.prototxt'
weight_path = './models/VGG_ILSVRC_19_layers.caffemodel'

optimizer = 'adam' # 还可以填 lbfgs
lr = 0.0001