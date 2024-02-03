## DeepAlchemy 接口文档


### def run(in_dict)

in_dict中内容：

gpu: int值，输入使用的gpu

modelname: str值，支持'resnet' 'vgg' 以及'mobilenet'

dataset: str值，支持'cifar100' 'cifar10' 'mnist'

epochs: int值, 训练轮数

init: str值，支持'normal' 'rand' 'large'

iternum: int值，建议取值3-6，最推荐4