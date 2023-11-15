import os
import argparse
import torch

from function.fairness.image import models
from function.fairness.image import utils
from function.fairness.image.metrics import *
from IOtool import IOtool

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',
                        default='cifar-s_baseline',
                        choices=[
                                 'cifar_color', # train on color images
                                 'cifar_gray', # train on gray images
                                 
                                 # cifar10-s color vs gray 
                                 'cifar-s_baseline', 
                                 'cifar-s_sampling',
                                 'cifar-s_domain_discriminative',
                                 'cifar-s_domain_independent',
                                 'cifar-s_uniconf_adv',
                                 'cifar-s_gradproj_adv',
                                 
                                 # cifar10-s cifar vs imagenet
                                 'cifar-i_baseline', 
                                 'cifar-i_sampling',
                                 'cifar-i_domain_discriminative',
                                 'cifar-i_domain_independent',
                                 
                                 # cifar10-s cifar vs 28 crop
                                 'cifar-c_28_baseline',
                                 'cifar-c_28_sampling',
                                 'cifar-c_28_domain_discriminative',
                                 'cifar-c_28_domain_independent',
                                 
                                 #cifar10-s cifar vs 16 downres
                                 'cifar-d_16_baseline',
                                 'cifar-d_16_sampling',
                                 'cifar-d_16_domain_discriminative',
                                 'cifar-d_16_domain_independent',
                                 
                                 #cifar10-s cifar vs 8 downres
                                 'cifar-d_8_baseline',
                                 'cifar-d_8_sampling',
                                 'cifar-d_8_domain_discriminative',
                                 'cifar-d_8_domain_independent',
                                 
                                 # celeba 
                                 'celeba_baseline', 
                                 'celeba_weighting',
                                 'celeba_domain_discriminative',
                                 'celeba_domain_independent',
                                 'celeba_uniconf_adv',
                                 'celeba_gradproj_adv',    
                                ],
                        )
    
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument("--save_folder", type=str, default='')
    parser.add_argument("--mdoel_path", type=str, default='')
    parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())
    opt = create_exerpiment_setting(opt)
    return opt

def create_exerpiment_setting(opt):
    opt['test_mode'] = True
    
    # common experiment setting
    if opt['experiment'].startswith('cifar'):
        opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
        # opt['device'] = IOtool.get_device()
        opt['print_freq'] = 50
        opt['batch_size'] = 128
        # opt['total_epochs'] = 200
        opt['total_epochs'] = 2
        # opt['save_folder'] = os.path.join('record/'+opt['experiment'], 
        #                                   opt['experiment_name'])
        if opt['save_folder'] == '' :
            opt['save_folder'] = os.path.join('output/cache/fairness/'+opt['experiment'], 
                                              opt['experiment_name'])
        utils.creat_folder(opt['save_folder'])
    
        optimizer_setting = {
            'optimizer': torch.optim.SGD,
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
        }
        opt['optimizer_setting'] = optimizer_setting
        opt['evaluate_func'] = cal_sl_metrics
        opt['metrics_list'] = sl_metrics
        
    elif opt['experiment'].startswith('celeba'):
        # opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
        opt['device'] = IOtool.get_device()
        opt['print_freq'] = 50
        opt['batch_size'] = 64
        opt['total_epochs'] = 50
        # opt['total_epochs'] = 2
        # opt['save_folder'] = os.path.join('record/'+opt['experiment'], 
        #                                   opt['experiment_name'])
        if opt['save_folder'] == '' :
            opt['save_folder'] = os.path.join('output/cache/fairness/'+opt['experiment'], 
                                          opt['experiment_name'])
        utils.creat_folder(opt['save_folder'])
        opt['output_dim'] = 39
        
        optimizer_setting = {
            'optimizer': torch.optim.Adam,
            'lr': 1e-4,
            'weight_decay': 0,
        }
        opt['optimizer_setting'] = optimizer_setting
        opt['dropout'] = 0.5
        data_setting = {
            'image_feature_path': 'dataset/fairness_data/data/celeba/celeba.h5py',
            'target_dict_path': 'dataset/fairness_data/data/celeba/labels_dict',
            'train_key_list_path': 'dataset/fairness_data/data/celeba/train_key_list',
            'dev_key_list_path': 'dataset/fairness_data/data/celeba/dev_key_list',
            'test_key_list_path': 'dataset/fairness_data/data/celeba/test_key_list',
            'subclass_idx_path': 'dataset/fairness_data/data/celeba/subclass_idx',
            'augment': True
        }
        opt['data_setting'] = data_setting
        opt['evaluate_func'] = cal_ml_metrics
        opt['metrics_list'] = ml_metrics
    
    # experiment-specific setting
    if opt['experiment'] == 'cifar_color':
        opt['output_dim'] = 10
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar_color_train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar_train_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar_gray_test_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_core.CifarModel(opt)
        
    elif opt['experiment'] == 'cifar_gray':
        opt['output_dim'] = 10
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar_gray_train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar_train_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar_gray_test_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_core.CifarModel(opt)
        
    ############ cifar color vs gray ##############
    
    elif opt['experiment'] == 'cifar-s_baseline':
        opt['output_dim'] = 10
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar_train_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar_gray_test_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_core.CifarModel(opt)
    
    elif opt['experiment'] == 'cifar-s_sampling':
        opt['output_dim'] = 10
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-s/p95.0/balanced_train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-s/p95.0/balanced_train_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar_gray_test_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_core.CifarModel(opt)
        
    elif opt['experiment'] == 'cifar-s_domain_discriminative':
        opt['output_dim'] = 20
        opt['prior_shift_weight'] = [1/5 if i%2==0 else 1/95 for i in range(10)] \
                                    + [1/95 if i%2==0 else 1/5 for i in range(10)]
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_2n_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar_gray_test_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_domain_discriminative.CifarDomainDiscriminative(opt)
        
    elif opt['experiment'] == 'cifar-s_domain_independent':
        opt['output_dim'] = 20
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_2n_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar_gray_test_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'domain_label_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_domain_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_domain_independent.CifarDomainIndependent(opt)
        
    elif opt['experiment'] == 'cifar-s_uniconf_adv':
        opt['output_dim'] = 10
        # opt['total_epochs'] = 500
        opt['total_epochs'] = 2
        opt['training_ratio'] = 3
        opt['alpha'] = 1.
        
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar_train_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar_gray_test_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'domain_label_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_domain_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        
        optimizer_setting = {
            'optimizer': torch.optim.Adam,
            'lr': 1e-4,
            'weight_decay': 3e-4,
        }
        opt['optimizer_setting'] = optimizer_setting
        
        model = models.cifar_uniconf_adv.CifarUniConfAdv(opt)
        
    elif opt['experiment'] == 'cifar-s_gradproj_adv':
        opt['output_dim'] = 10
        # opt['total_epochs'] = 500
        opt['total_epochs'] = 2
        opt['training_ratio'] = 3
        opt['alpha'] = 1.
        
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar_train_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar_gray_test_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'domain_label_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_domain_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        
        optimizer_setting = {
            'optimizer': torch.optim.Adam,
            'lr': 1e-4,
            'weight_decay': 3e-4,
        }
        opt['optimizer_setting'] = optimizer_setting
        
        model = models.cifar_gradproj_adv.CifarGradProjAdv(opt)
            
    ############ cifar vs imagenet ##############
    
    elif opt['experiment'] == 'cifar-i_baseline':
        opt['output_dim'] = 10
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-i/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar_train_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-i/cinic_test_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_core.CifarModel(opt)
        
    elif opt['experiment'] == 'cifar-i_sampling':
        opt['output_dim'] = 10
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-i/balanced_train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-i/balanced_train_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-i/cinic_test_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_core.CifarModel(opt)
                
    elif opt['experiment'] == 'cifar-i_domain_discriminative':
        opt['output_dim'] = 20
        opt['prior_shift_weight'] = [1/5 if i%2==0 else 1/95 for i in range(10)] \
                                    + [1/95 if i%2==0 else 1/5 for i in range(10)]
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-i/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_2n_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-i/cinic_test_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_domain_discriminative.CifarDomainDiscriminative(opt)
        
    elif opt['experiment'] == 'cifar-i_domain_independent':
        opt['output_dim'] = 20
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-i/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_2n_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-i/cinic_test_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_domain_independent.CifarDomainIndependent(opt)
        
    ############ cifar vs 28 cropped ############## 
    
    elif opt['experiment'] == 'cifar-c_28_baseline':
        opt['output_dim'] = 10
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-c/c28/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar_train_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-c/c28/test_crop_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_core.CifarModel(opt)
        
    elif opt['experiment'] == 'cifar-c_28_sampling':
        opt['output_dim'] = 10
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-c/c28/balanced_train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-c/c28/balanced_train_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-c/c28/test_crop_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_core.CifarModel(opt)
                
    elif opt['experiment'] == 'cifar-c_28_domain_discriminative':
        opt['output_dim'] = 20
        opt['prior_shift_weight'] = [1/5 if i%2==0 else 1/95 for i in range(10)] \
                                    + [1/95 if i%2==0 else 1/5 for i in range(10)]
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-c/c28/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_2n_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-c/c28/test_crop_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_domain_discriminative.CifarDomainDiscriminative(opt)
        
    elif opt['experiment'] == 'cifar-c_28_domain_independent':
        opt['output_dim'] = 20
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-c/c28/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_2n_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-c/c28/test_crop_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_domain_independent.CifarDomainIndependent(opt)
    
    ############ cifar vs 16 downres ##############   
    
    elif opt['experiment'] == 'cifar-d_16_baseline':
        opt['output_dim'] = 10
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-d/d16/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar_train_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-d/d16/test_downsamp_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_core.CifarModel(opt)
        
    elif opt['experiment'] == 'cifar-d_16_sampling':
        opt['output_dim'] = 10
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-d/d16/balanced_train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-d/d16/balanced_train_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-d/d16/test_downsamp_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_core.CifarModel(opt)
                
    elif opt['experiment'] == 'cifar-d_16_domain_discriminative':
        opt['output_dim'] = 20
        opt['prior_shift_weight'] = [1/5 if i%2==0 else 1/95 for i in range(10)] \
                                    + [1/95 if i%2==0 else 1/5 for i in range(10)]
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-d/d16/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_2n_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-d/d16/test_downsamp_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_domain_discriminative.CifarDomainDiscriminative(opt)
        
    elif opt['experiment'] == 'cifar-d_16_domain_independent':
        opt['output_dim'] = 20
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-d/d16/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_2n_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-d/d16/test_downsamp_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_domain_independent.CifarDomainIndependent(opt)
    
    ############ cifar vs 8 downres ##############  
    
    elif opt['experiment'] == 'cifar-d_8_baseline':
        opt['output_dim'] = 10
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-d/d8/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar_train_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-d/d8/test_downsamp_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_core.CifarModel(opt)
        
    elif opt['experiment'] == 'cifar-d_8_sampling':
        opt['output_dim'] = 10
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-d/d8/balanced_train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-d/d8/balanced_train_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-d/d8/test_downsamp_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_core.CifarModel(opt)
                
    elif opt['experiment'] == 'cifar-d_8_domain_discriminative':
        opt['output_dim'] = 20
        opt['prior_shift_weight'] = [1/5 if i%2==0 else 1/95 for i in range(10)] \
                                    + [1/95 if i%2==0 else 1/5 for i in range(10)]
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-d/d8/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_2n_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-d/d8/test_downsamp_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_domain_discriminative.CifarDomainDiscriminative(opt)
        
    elif opt['experiment'] == 'cifar-d_8_domain_independent':
        opt['output_dim'] = 20
        data_setting = {
            'train_data_path': 'dataset/fairness_data/data/cifar-d/d8/train_imgs',
            'train_label_path': 'dataset/fairness_data/data/cifar-s/p95.0/train_2n_labels',
            'test_color_path': 'dataset/fairness_data/data/cifar_color_test_imgs',
            'test_gray_path': 'dataset/fairness_data/data/cifar-d/d8/test_downsamp_imgs',
            'test_label_path': 'dataset/fairness_data/data/cifar_test_labels',
            'augment': True
        }
        opt['data_setting'] = data_setting
        model = models.cifar_domain_independent.CifarDomainIndependent(opt)
    
    ############ celeba ##############   
    
    elif opt['experiment'] == 'celeba_baseline':
        model = models.celeba_core.CelebaModel(opt)
        
    elif opt['experiment'] == 'celeba_weighting':
        model = models.celeba_weighting.CelebaWeighting(opt)
        
    elif opt['experiment'] == 'celeba_domain_discriminative':
        opt['output_dim'] = 78
        model = models.celeba_domain_discriminative.CelebaDomainDiscriminative(opt)
    
    elif opt['experiment'] == 'celeba_domain_independent':
        opt['output_dim'] = 78
        model = models.celeba_domain_independent.CelebaDomainIndependent(opt)
    
    elif opt['experiment'] == 'celeba_uniconf_adv':
        opt['training_ratio'] = 3
        opt['alpha'] = 1.
        model = models.celeba_uniconf_adv.CelebaUniConfAdv(opt)
        
    elif opt['experiment'] == 'celeba_gradproj_adv':
        opt['training_ratio'] = 3
        opt['alpha'] = 1.
        model = models.celeba_gradproj_adv.CelebaGradProjAdv(opt)
        
    return model, opt