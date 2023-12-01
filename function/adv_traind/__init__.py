import json,torch, os, shutil
from .FeatureScatter.models.resnet import ResNet
from .FeatureScatter.models.lenet import LeNet
from .FeatureScatter.models.wideresnet import WideResNet
from .FeatureScatter.models.vggnet import VGG
from .FeatureScatter.LoadModel import RobustEnhance
from .FeatureScatter.AttackModel import RobustTest
from .FeatureScatter.normal_train import train_normal_model

def run_featurescatter(dataset, modelname, attack_method, evaluate_methods, 
                       lr, batch_size, max_epoch, decay_epoch, decay_rate, out_path, logging=None):
    # 数据集和模型加载
    logging.info('Dataset preparing......')
    if dataset == 'cifar100':
        n_class = 100
    else:
        n_class = 10
    image_size = 32
    if modelname == 'resnet':
        model = ResNet(50, n_class)
    elif modelname == 'lenet':
        model = LeNet(n_class)
    elif modelname == 'wideresnet':
        model = WideResNet(depth=28, num_classes=n_class, widen_factor=10)
    elif modelname == 'vgg':
        model = VGG(16, n_class)
    logging.info('Dataset prepared......')
    
    # 构建参数dict
    args_dict = {
        'dataset': dataset, 
        'modelname':modelname,
        'init_model_pass': 'latest',
        'model_dir': out_path,
        'resume': True,
        'lr': lr,
        'adv_mode': 'featurescatter',
        
        'attack': True,
        'attack_method': attack_method,
        'attack_method_list': evaluate_methods, #'natural-FGSM-PGD-CW'
        
        'max_epoch': max_epoch,
        'save_epochs': 20,
        'decay_epoch1': decay_epoch,
        'decay_epoch2': 90,
        'decay_rate': decay_rate,
        'batch_size_train': batch_size,
        'batch_size_test': batch_size,
        'momentum': 0.9,
        'weight_decay': 2e-4,
        'log_step': 50,
        'num_classes': n_class,
        'image_size': image_size,
    }
    logging.info('start normal train......')
    results = dict()
    results['normal_train'] = train_normal_model(model=model, args_dict=args_dict, logging=logging)
    logging.info('end normal train......')
    logging.info(f'start feature scatter train......')
    RobustEnhance(model, args_dict, logging=logging)
    logging.info('start test......')
    results['feature_scatter'] = RobustTest(model, args_dict)
    logging.info('finish feature scatter robust training......')
    return results
        
from .SEAT.seat import seatrain, seatest

def run_seat(dataset, modelname, lr, epsilon, max_epoch, evaluate_method, out_path, logging=None):
    
    logging.info('Start SEAT running......')
    
    args_dict = {
        'epochs': max_epoch, 
        'arch': modelname,
        'lr': lr,
        'loss_fn': 'cent',
        'epsilon': epsilon,
        'num_steps': 10,
        'step_size': 0.007,
        'resume': False,  # test:True train:False
        'out_dir': out_path,
        'ablation': '',
        'evaluate_method': evaluate_method}
    
    if not os.path.exists(os.path.join("output/cache/SEAT",modelname,'bestpoint.pth.tar')) or not os.path.exists(os.path.join("output/cache/SEAT",modelname,'ema_bestpoint.pth.tar')):
        logging.info('Pretrained model not existed......')
        if os.path.exists(os.path.join("output/cache/SEAT",modelname)):
            shutil.rmtree(os.path.join("output/cache/SEAT",modelname))
        logging.info('Model training......')
        seatrain(args_dict)
        shutil.copytree(out_path, os.path.join("output/cache/SEAT",modelname))
        logging.info('Model training finished......')
    else:
        shutil.copytree(os.path.join("output/cache/SEAT",modelname),out_path)
    logging.info('Model testing......')
    res = seatest(args_dict)
    logging.info('Finish SEAT testing......')
    return res
    
    
    
from .RiFT.test import start_evaluate
def run_rift(dataset, model, attack_method, evaluate_methods,  train_epoch, at_epoch, batchsize, out_path, logging=None, device=None):
    logging.info("Start RiFT training ")
    res = start_evaluate(ori_train_epoch=train_epoch, at_method=attack_method.lower(), at_eps=4, at_epoch=at_epoch, dataset=dataset.lower(), model_name=model.lower(),
                   at_train_epoch=at_epoch, rift_epoch=10, batch_size=batchsize,
                   evaluate_methods=[item.lower() for item in evaluate_methods], save_path=out_path, device="cuda")
    logging.info("End RiFT training ")
    return res