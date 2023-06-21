import os
import json
import pickle
import random
import copy
import h5py
import numpy as np
from PIL import Image
from datetime import datetime

cifar_color_class = {'automobile', 'cat', 'dog', 'horse', 'truck'}
cifar_gray_class = {'airplane', 'bird', 'deer', 'frog', 'ship'}
cifar_class_dict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                    'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

def rgb_to_grayscale(img):
    """Convert image to gray scale"""
    
    pil_img = Image.fromarray(img)
    pil_gray_img = pil_img.convert('L')
    np_gray_img = np.array(pil_gray_img, dtype=np.uint8)
    np_gray_img = np.dstack([np_gray_img, np_gray_img, np_gray_img])
    
    return np_gray_img

def down_res(img, size=(16, 16)):
    """Downsize then upsize to have a lower resolution image"""
    
    pil_img = Image.fromarray(img)
    resize_img = pil_img.resize(size)
    np_img = np.array(resize_img.resize((32, 32)), dtype=np.uint8)
    
    return np_img

def center_crop_upres(img, cropsize):
    """Crop the image and then upsize"""
    
    pil_img = Image.fromarray(img)
    size = pil_img.size[0]
    cropout_size = (size - cropsize) // 2
    crop_img = pil_img.crop((cropout_size, cropout_size,
                             size-cropout_size, size-cropout_size))
    np_img = np.array(crop_img.resize((32, 32)), dtype=np.uint8)
    return np_img

def create_cifar_data():
    """Generate dataset for all cifar experiments"""
    
    # Generate cifar color vs gray
    train_imgs = []
    train_labels = []
    
    for i in range(1, 6):
        with open('data/cifar10/data_batch_{}'.format(i), 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
        
        train_imgs.append(batch['data'])
        train_labels.extend(batch['labels'])
        
    train_imgs = np.vstack(train_imgs).reshape(-1, 3, 32, 32)
    train_imgs = train_imgs.transpose((0, 2, 3, 1))
    
    with open('data/cifar_train_labels', 'wb') as f:
        pickle.dump(train_labels, f)
    with open('data/cifar_color_train_imgs', 'wb') as f:
        pickle.dump(train_imgs, f)
        
    cifar_gray_train = train_imgs.copy()
    for i in range(cifar_gray_train.shape[0]):
        cifar_gray_train[i] = rgb_to_grayscale(cifar_gray_train[i])

    with open('data/cifar_gray_train_imgs', 'wb') as f:
        pickle.dump(cifar_gray_train, f)
    
    with open('data/cifar10/test_batch', 'rb') as f:
        test_batch = pickle.load(f, encoding='latin1')
        
    test_imgs = test_batch['data'].reshape(-1, 3, 32, 32)
    test_imgs = test_imgs.transpose((0, 2, 3, 1))
    test_labels = test_batch['labels']
    
    with open('data/cifar_test_labels', 'wb') as f:
        pickle.dump(test_labels, f)
    with open('data/cifar_color_test_imgs', 'wb') as f:
        pickle.dump(test_imgs, f)
        
    cifar_gray_test = test_imgs.copy()
    for i in range(cifar_gray_test.shape[0]):
        cifar_gray_test[i] = rgb_to_grayscale(cifar_gray_test[i])
    
    with open('data/cifar_gray_test_imgs', 'wb') as f:
        pickle.dump(cifar_gray_test, f)
        
    with open('data/cifar_test_two_n_labels', 'wb') as f:
        pickle.dump([label+10 for label in test_labels], f)
        
    # Generate data wtih different skew level
    for imbalance_p in [0.75, 0.8, 0.875, 0.9, 0.95, 0.96, 0.975, 0.98, 0.99]:
        create_cifar_s_data(train_imgs, train_labels, imbalance_p)
        
    # Generate cifar vs imagenet data 
    create_cifar_i_data()
    create_cifars_domain_label('./data/cifar-s/p95.0')
    
    with open('data/cifar-s/p95.0/domain_idx', 'rb') as f:
        domain_idx_dict = pickle.load(f)
    with open('data/cifar-s/p95.0/train_imgs', 'rb') as f:
        train_imgs = pickle.load(f)
    with open('data/cifar_train_labels', 'rb') as f:
        train_labels = pickle.load(f)
    with open('data/cifar_color_test_imgs', 'rb') as f:
        test_imgs = pickle.load(f)
    
    # Generate cifar vs low-res data
    create_cifar_d_data(train_imgs, train_labels, domain_idx_dict,
                        test_imgs, 16)
    create_cifar_d_data(train_imgs, train_labels, domain_idx_dict,
                        test_imgs, 8)
    
    # Generate cifar vs cropped data
    create_cifar_c_data(train_imgs, train_labels, domain_idx_dict,
                        test_imgs, 28)
    
def create_cifar_s_data(train_imgs, train_labels, imbalance_p):
    """Generate cifar-s data with a given skew level"""
    
    random.seed(42)
    if not os.path.exists('data/cifar-s/p{}'.format(imbalance_p*100)):
        os.makedirs('data/cifar-s/p{}'.format(imbalance_p*100))
    
    color_class_idx_dict = {}
    gray_class_idx_dict = {}
    
    total_num_per_class = 5000
    sample_num = int(total_num_per_class * imbalance_p)
    color_class_id = set(cifar_class_dict[name] for name in cifar_color_class)
    
    class_idx_dict = {}
    for class_id in cifar_class_dict.values():
        class_idx_dict[class_id] = [idx 
                    for idx, c in enumerate(train_labels) if c==class_id]
    
    for i in cifar_class_dict.values():
        if i in color_class_id:
            color_img_idx = random.sample(class_idx_dict[i], sample_num)
            color_class_idx_dict[i] = color_img_idx
            gray_class_idx_dict[i] = list(set(class_idx_dict[i]) - set(color_img_idx))
        else:
            gray_img_idx = random.sample(class_idx_dict[i], sample_num)
            gray_class_idx_dict[i] = gray_img_idx
            color_class_idx_dict[i] = list(set(class_idx_dict[i]) - set(gray_img_idx))
            
    domain_idx_dict = {'color_idx': color_class_idx_dict,
                       'gray_idx': gray_class_idx_dict}
    with open('data/cifar-s/p{}/domain_idx'.format(imbalance_p*100), 'wb') as f:
        pickle.dump(domain_idx_dict, f)
    
    weight_list = [-1]*50000
    for i in range(10):
        total_weight = len(color_class_idx_dict[i]) + len(gray_class_idx_dict[i])
        for idx in color_class_idx_dict[i]:
            weight_list[idx] = total_weight / 2 / len(color_class_idx_dict[i])
        for idx in gray_class_idx_dict[i]:
            weight_list[idx] = total_weight / 2 / len(gray_class_idx_dict[i])
    with open('data/cifar-s/p{}/sample_weight'.format(imbalance_p*100), 'wb') as f:
        pickle.dump(weight_list, f)
        
    train_skew = train_imgs.copy()
    for idx_list in gray_class_idx_dict.values():
        for i in idx_list:
            train_skew[i] = rgb_to_grayscale(train_skew[i])
    
    with open('data/cifar-s/p{}/train_imgs'.format(imbalance_p*100), 'wb') as f:
        pickle.dump(train_skew, f)
        
    two_n_labels = train_labels.copy()
    for i in cifar_class_dict.values():
        for idx in gray_class_idx_dict[i]:
            assert two_n_labels[idx] == i, 'Label mismatch...'
            two_n_labels[idx] += 10
            
    with open('data/cifar-s/p{}/train_2n_labels'.format(imbalance_p*100), 'wb') as f:
        pickle.dump(two_n_labels, f)
        
    create_balanced_data(train_skew, train_labels, domain_idx_dict, 
                         'cifar-s/p{}'.format(imbalance_p*100))
        
def create_balanced_data(train_imgs, train_labels, domain_idx_dict, save_folder_name='cifar-s'):
    """Oversampling to create a balanced training set"""
    
    added_balancing_imgs = []
    added_balancing_labels = []
    balanced_color_class_idx_dict = copy.deepcopy(domain_idx_dict['color_idx'])
    balanced_gray_class_idx_dict = copy.deepcopy(domain_idx_dict['gray_idx'])
    check_labels = []
    
    added_idx = len(train_labels)
    for i in cifar_class_dict.values():
        if len(domain_idx_dict['color_idx'][i]) < len(domain_idx_dict['gray_idx'][i]):
            for idx in domain_idx_dict['color_idx'][i]:
                duplicate_num = (len(domain_idx_dict['gray_idx'][i])
                                // len(domain_idx_dict['color_idx'][i])) \
                                - 1
                for j in range(duplicate_num):
                    added_balancing_imgs.append(train_imgs[idx].copy())
                    added_balancing_labels.append(i)
                    check_labels.append(train_labels[idx])
                    balanced_color_class_idx_dict[i].append(added_idx)
                    added_idx += 1
        else:
            for idx in domain_idx_dict['gray_idx'][i]:
                duplicate_num = (len(domain_idx_dict['color_idx'][i])
                                // len(domain_idx_dict['gray_idx'][i])) \
                                - 1
                for j in range(duplicate_num):
                    added_balancing_imgs.append(train_imgs[idx].copy())
                    added_balancing_labels.append(i)
                    check_labels.append(train_labels[idx])
                    balanced_gray_class_idx_dict[i].append(added_idx)
                    added_idx += 1
    
    assert check_labels == added_balancing_labels
    with open('data/{}/balanced_domain_idx'.format(save_folder_name), 'wb') as f:
        pickle.dump({'color_idx': balanced_color_class_idx_dict, 
                     'gray_idx': balanced_gray_class_idx_dict}, f)
        
    train_balanced_imgs = np.vstack((train_imgs, np.stack(added_balancing_imgs)))
    train_balanced_lables = train_labels + added_balancing_labels
    with open('data/{}/balanced_train_imgs'.format(save_folder_name), 'wb') as f:
        pickle.dump(train_balanced_imgs, f)
    with open('data/{}/balanced_train_labels'.format(save_folder_name), 'wb') as f:
        pickle.dump(train_balanced_lables, f)
        
def create_cifar_d_data(train_imgs, train_labels, domain_idx_dict, 
                        test_imgs, down_size):
    """Create cifar vs low-res data"""
    
    if not os.path.exists('data/cifar-d/d{}'.format(down_size)):
        os.makedirs('data/cifar-d/d{}'.format(down_size))
    
    train_downsamp = train_imgs.copy()
    for idx_list in domain_idx_dict['gray_idx'].values():
        for i in idx_list:
            train_downsamp[i] = down_res(train_downsamp[i], (down_size,down_size))
    with open('data/cifar-d/d{}/train_imgs'.format(down_size), 'wb') as f:
        pickle.dump(train_downsamp, f)
    
    test_downsamp = test_imgs.copy()
    for i in range(test_downsamp.shape[0]):
        test_downsamp[i] = down_res(test_downsamp[i], (down_size, down_size))
    with open('data/cifar-d/d{}/test_downsamp_imgs'.format(down_size), 'wb') as f:
        pickle.dump(test_downsamp, f)  
        
    create_balanced_data(train_downsamp, train_labels, domain_idx_dict, 
                         'cifar-d/d{}'.format(down_size))
    
def create_cifar_c_data(train_imgs, train_labels, domain_idx_dict,
                        test_imgs, crop_size):
    """Create cifar vs cropped data"""
    
    if not os.path.exists('data/cifar-c/c{}'.format(crop_size)):
        os.makedirs('data/cifar-c/c{}'.format(crop_size))
    
    train_crop = train_imgs.copy()
    for idx_list in domain_idx_dict['gray_idx'].values():
        for i in idx_list:
            train_crop[i] = center_crop_upres(train_crop[i], crop_size)
    with open('data/cifar-c/c{}/train_imgs'.format(crop_size), 'wb') as f:
        pickle.dump(train_crop, f)
    
    test_crop = test_imgs.copy()
    for i in range(test_crop.shape[0]):
        test_crop[i] = center_crop_upres(test_crop[i], crop_size)
    with open('data/cifar-c/c{}/test_crop_imgs'.format(crop_size), 'wb') as f:
        pickle.dump(test_crop, f)  
        
    create_balanced_data(train_crop, train_labels, domain_idx_dict, 
                         'cifar-c/c{}'.format(crop_size))
    
def create_cifar_i_data():
    """Create cifar vs imagenet data"""
    
    if not os.path.exists('data/cifar-i'):
        os.makedirs('data/cifar-i')
    
    cinic_path = 'data/cinic'
    cinic_train_images = {}
    cinic_test_images = {}
    for cls_name in cifar_color_class:
        cls_idx = cifar_class_dict[cls_name]
        cinic_train_images[cls_idx] = []
        img_num = 0
        for i, filename in enumerate(os.listdir('data/cinic/train/'+cls_name)):
            file_path = os.path.join('data/cinic/train/'+cls_name, filename)
            if ('cifar' not in filename) and (Image.open(file_path).mode == 'RGB'):
                cinic_train_images[cls_idx].append(np.array(Image.open(file_path), dtype=np.uint8))
                img_num += 1
                if img_num == 250:
                    break
                
        cinic_test_images[cls_idx] = []
        img_num = 0
        for i, filename in enumerate(os.listdir('data/cinic/test/'+cls_name)):
            file_path = os.path.join('data/cinic/test/'+cls_name, filename)
            if ('cifar' not in filename) and (Image.open(file_path).mode == 'RGB'):
                cinic_test_images[cls_idx].append(np.array(Image.open(file_path), dtype=np.uint8))
                img_num += 1
                if img_num == 1000:
                    break
                
    for cls_name in cifar_gray_class:
        cls_idx = cifar_class_dict[cls_name]
        cinic_train_images[cls_idx] = []
        img_num = 0
        for i, filename in enumerate(os.listdir('data/cinic/train/'+cls_name)):
            file_path = os.path.join('data/cinic/train/'+cls_name, filename)
            if ('cifar' not in filename) and (Image.open(file_path).mode == 'RGB'):
                cinic_train_images[cls_idx].append(np.array(Image.open(file_path), dtype=np.uint8))
                img_num += 1
                if img_num == 4750:
                    break      
                
        cinic_test_images[cls_idx] = []
        img_num = 0
        for i, filename in enumerate(os.listdir('data/cinic/test/'+cls_name)):
            file_path = os.path.join('data/cinic/test/'+cls_name, filename)
            if ('cifar' not in filename) and (Image.open(file_path).mode == 'RGB'):
                cinic_test_images[cls_idx].append(np.array(Image.open(file_path), dtype=np.uint8))
                img_num += 1
                if img_num == 1000:
                    break
                
    with open('data/cifar-s/p95.0/domain_idx', 'rb') as f:
        domain_idx = pickle.load(f)
    with open('data/cifar-s/p95.0/train_imgs', 'rb') as f:
        train_imgs = pickle.load(f)
    with open('data/cifar_train_labels', 'rb') as f:
        train_labels = pickle.load(f)
    with open('data/cifar_color_test_imgs', 'rb') as f:
        test_imgs = pickle.load(f)
    with open('data/cifar_test_labels', 'rb') as f:
        test_labels = pickle.load(f)
    
    for cls_idx, image_idx_list in domain_idx['gray_idx'].items():
        assert len(image_idx_list) == len(cinic_train_images[cls_idx])
        for i, image_idx in enumerate(image_idx_list):
            train_imgs[image_idx] = cinic_train_images[cls_idx][i]
    
    with open('data/cifar-i/train_imgs', 'wb') as f:
        pickle.dump(train_imgs, f)
        
    for i, cls_idx in enumerate(test_labels):
        test_imgs[i] = cinic_test_images[cls_idx].pop()
        
    with open('data/cifar-i/cinic_test_imgs', 'wb') as f:
        pickle.dump(test_imgs, f)
        
    create_balanced_data(train_imgs, train_labels, domain_idx, save_folder_name='cifar-i')
    
def create_celeba_data(image_path):   
    """Create dataset for celeba experiments"""
    
    if not os.path.exists('data/celeba'):
        os.makedirs('data/celeba')
    
    feature_file = h5py.File('data/celeba/celeba.h5py', "w")
    for filename in os.listdir(image_path):
        feature_file.create_dataset(filename, 
            data=np.asarray(Image.open(os.path.join(image_path, filename)).convert('RGB')))
    feature_file.close()
    
    with open('data/celeba/Anno/list_attr_celeba.txt', 'r') as f:
        lines = f.readlines()
        
    attr_list = lines[1].strip().split()
    attr_idx_dict = {attr: i for i, attr in enumerate(attr_list)}
    labels_dict = {}
    for line in lines[2:]:
        line = line.strip().split()
        key = line[0]
        attr = line[1:]
        attr.append(attr.pop(attr_idx_dict['Male']))
        attr = np.array(attr).astype(int)
        attr = (attr + 1) / 2
        labels_dict[key] = attr.copy()
        
    with open('data/celeba/labels_dict', 'wb') as f:
        pickle.dump(labels_dict, f)
    
    with open('data/celeba/Eval/list_eval_partition.txt', 'r') as f:
        split_lines = f.readlines()
        
    train_list = []
    dev_list = []
    test_list = []
    for i, line in enumerate(split_lines):
        line = line.strip().split()
        if line[1] == '0':
            train_list.append(line[0])
        elif line[1] == '1':
            dev_list.append(line[0])
        elif line[1] == '2':
            test_list.append(line[0])
        else:
            print('error')
            break
            
    with open('data/celeba/train_key_list', 'wb') as f:
        pickle.dump(train_list, f)
    with open('data/celeba/dev_key_list', 'wb') as f:
        pickle.dump(dev_list, f)
    with open('data/celeba/test_key_list', 'wb') as f:
        pickle.dump(test_list, f)
    
    subclass_idx = list(set(range(39)) - {0,16,21,29,37})
    with open('data/celeba/subclass_idx', 'wb') as f:
        pickle.dump(subclass_idx, f)

def create_cifars_domain_label(data_folder):
    """Generate domain label for adversarial experiments"""
    
    with open(os.path.join(data_folder, 'domain_idx'), 'rb') as f:
        domain_idx = pickle.load(f)
    with open(os.path.join(data_folder, 'train_imgs'), 'rb') as f:
        train_imgs = pickle.load(f)
        
    domain_labels = [0]*train_imgs.shape[0]
    for gray_idx_list in domain_idx['gray_idx'].values():
        for idx in gray_idx_list:
            domain_labels[idx] = 1
            
    with open(os.path.join(data_folder, 'train_domain_labels'), 'wb') as f:
        pickle.dump(domain_labels, f)
        
if __name__ == '__main__':
    # print('Preparing cifar experiment data')
    # create_cifar_data()
    print('Preparing celeba experiment data')
    create_celeba_data('./data/celeba/images')
    print('Finshed')

    