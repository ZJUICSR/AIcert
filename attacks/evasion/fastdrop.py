# from __future__ import absolute_import, division, print_function, unicode_literals
from tqdm import tqdm
from PIL import Image
# import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import torch
import torchvision.transforms as transforms
import logging
from typing import Optional, TYPE_CHECKING, List
import numpy as np
from scipy.fftpack import idct
from tqdm.auto import trange
from attacks.attack import EvasionAttack
from estimators.estimator import BaseEstimator, NeuralNetworkMixin
from estimators.classification.classifier import ClassifierMixin
from attacks.config import MY_NUMPY_DTYPE
from attacks.utils import is_probability
if TYPE_CHECKING:
    from attacks.utils import CLASSIFIER_TYPE
from attacks.utils import check_and_transform_label_format, projection, random_sphere, is_probability, get_labels_np_array, get_label_conf

class Fastdrop(EvasionAttack):

    attack_params = EvasionAttack.attack_params + [
        "epsilon",
        "batch_size",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin, NeuralNetworkMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        batch_size = 1,
    ):
        super().__init__(estimator=classifier)
        self.batch_size = batch_size
        self.l2_norm_thres = 5.0
        self.square_max_num = 28
        # self.device = self.estimator.model.device
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.trans1 = transforms.ToTensor()
    
        # def datatrans(data) -> np.ndarray:
        #     data = np.transpose(data, (0, 3, 1, 2)).astype(np.float32)
        #     return data
        
        # def redatatrans(data) -> np.ndarray:
        #     data = np.transpose(data, (0, 2, 3, 1)).astype(np.float32)
        #     return data
    
        # self.datatrans = datatrans
        # self.redatatrans = redatatrans
    
    def square_avg(self, freq:np.ndarray, index:int):
        rank1 = np.sum(freq[index, index:self.square_max_num-index, :])
        rank2 = np.sum(freq[self.square_max_num-1-index, index:self.square_max_num-index, :])
        col1 = np.sum(freq[index+1:self.square_max_num-1-index, index, :])
        col2 = np.sum(freq[index+1:self.square_max_num-1-index, self.square_max_num-1-index, :])
        num = 4*(self.square_max_num - 2*index) - 2

        return (rank1+rank2+col1+col2) / float(num)

    def square_zero(self, freq:np.ndarray, index:int):
        freq_modified = freq.copy()
        freq_modified[index, index:self.square_max_num-index, :] = 0
        freq_modified[self.square_max_num-1-index, index:self.square_max_num-index, :] = 0
        freq_modified[index:self.square_max_num-index:, index, :] = 0
        freq_modified[index:self.square_max_num-index, self.square_max_num-1-index, :] = 0

        return freq_modified

    def square_recover(self, freq_modified:np.ndarray, freq_ori:np.ndarray, index:int):
        freq_modified[index, index:self.square_max_num-index, :] = freq_ori[index, index:self.square_max_num-index, :]
        freq_modified[self.square_max_num-1-index, index:self.square_max_num-index, :] = freq_ori[self.square_max_num-1-index, index:self.square_max_num-index, :]
        freq_modified[index:self.square_max_num-index:, index, :] = freq_ori[index:self.square_max_num-index:, index, :]
        freq_modified[index:self.square_max_num-index, self.square_max_num-1-index, :] = freq_ori[index:self.square_max_num-index, self.square_max_num-1-index, :]

        return freq_modified

    # input为img，output为输入到神经网络的预处理之后的图片square_avg
    def generate_by_one_img(self, x:np.ndarray):
        self.square_max_num = x.shape[2]
        # 分割路径和文件名
        # path1, path2 = os.path.split(file_path)
        # 取得保存路径
        # save_path = path1 + '_adv/' + path2
        query_num = 0
        ori_img = np.transpose(x, (1,2,0))
        # 转化为tensor，并且添加维度
        # img = x.unsqueeze(0)
        x = np.expand_dims(x, axis=0)
        # img = img.to(device)
        # 查询网络
        query_num += 1
        # with torch.no_grad():
        #     out = net(img)
        #     _, ori_label = torch.max(out, dim=1)
        # ori_label = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size)).astype(int)[0]
        _, ori_label = get_label_conf(self.estimator.predict(x, batch_size=self.batch_size))

        # DFT
        # fft to original numpy image
        # 在原始图片上进行傅里叶变换
        freq = np.fft.fft2(ori_img, axes=(0, 1))
        # 保留傅里叶变换后的数据
        freq_ori = freq.copy()
        #
        freq_ori_m = np.abs(freq_ori)
        freq_abs = np.abs(freq)
        # 分块数目
        num_block = int(self.square_max_num/2)
        block_sum = np.zeros(num_block)
        for i in range(num_block):
            block_sum[i] = self.square_avg(freq_abs, i)

        # ordered index
        block_sum_ind = np.argsort(block_sum)
        # block_sum_ind = block_sum_ind[::-1]
        block_sum_ind_flag = np.zeros(num_block)

        # with open(log_file, 'a') as f:
        #     print('second stage!!!', file=f)
        img_save = None
        range_0 = [1,2]
        range_1 = range(3, num_block+1, 1)
        mags = range_0 + list(range_1)
        freq_sec_stage = freq.copy()
        freq_sec_stage_m = np.abs(freq_sec_stage)  
        freq_sec_stage_p = np.angle(freq_sec_stage)  
        mag_start = 0
        for mag in mags:
            for i in range(mag_start, mag):
                ind = block_sum_ind[i]
                freq_sec_stage_m = self.square_zero(freq_sec_stage_m, ind)
                freq_sec_stage = freq_sec_stage_m * np.e ** (1j * freq_sec_stage_p)

            img_adv = np.abs(np.fft.ifft2(freq_sec_stage, axes=(0, 1)))
            img_adv = np.clip(img_adv, 0, 1)  
            # img_adv = img_adv.astype('uint8')
            img_save = img_adv.copy()
            # img_adv = trans1(img_adv).unsqueeze(0)
            # img_adv = norm(img_adv)
            # img_adv = img_adv.to(device)
            query_num += 1
            # with torch.no_grad():
            #     out = net(img_adv)
            #     _, adv_label = torch.max(out, dim=1)
            # adv_label = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size)).astype(int)[0]
            _, adv_label = get_label_conf(self.estimator.predict(np.expand_dims(np.transpose(img_adv,(2,0,1)), axis=0).astype(MY_NUMPY_DTYPE), batch_size=self.batch_size))

            mag_start = mag
            if ori_label != adv_label:
                # print('max num success')
                # print('%d block' % (mag))
                # print('l2_norm: ', torch.norm((ori_img.squeeze()) - (img_adv.squeeze()), p=2).item())
                # print('linf_norm: ', torch.norm((ori_img.squeeze()) - (img_adv.squeeze()), p=np.inf).item())
                l2_norm = np.linalg.norm(ori_img.reshape(-1) - img_adv.reshape(-1), ord=2)
                # linf_norm = np.linalg.norm(ori_img - img_adv, ord=np.inf)
                if l2_norm.item() < self.l2_norm_thres:
                    # with open(log_file, 'a') as f:
                    #     print('%d block success' % (mag), file=f)
                        
                    img_save = img_save
                    # img_save.save(save_path)
                    # with open(log_file, 'a') as f:
                    #     print('query number: ', query_num, file=f)
                    #     print('l2_norm: ', torch.norm((img.squeeze()) - (img_adv.squeeze()), p=2).item(), file=f)
                    #     print('linf_norm: ', torch.norm((img.squeeze()) - (img_adv.squeeze()), p=np.inf).item(), file=f)
                else:
                    pass
                    # with open(log_file, 'a') as f:
                    #     print('success adv: ', mag_start, file=f)
                    #     print('l2_norm: ', torch.norm((img.squeeze()) - (img_adv.squeeze()), p=2).item(), file=f)
                    #     print('linf_norm: ', torch.norm((img.squeeze()) - (img_adv.squeeze()), p=np.inf).item(), file=f)

                break
            else:
                if mag == mags[-1]:
                    return x[0]


        # get adv example
        # with open(log_file, 'a') as f:
        #     print('third stage!!!', file=f)
        # img_save = None
        img_temp = img_save
        # max_i = -1
        max_i = mag_start - 1
        block_sum_ind_flag[:max_i+1] = 1
        # print('max_i: ', max_i)
        # freq_m = np.abs(freq)
        freq_m = freq_sec_stage_m
        freq_p = np.angle(freq)
        

        # optimize the adv example
        optimize_block = 0
        l2_norm = torch.tensor(0)
        linf_norm = torch.tensor(0)
        # with open(log_file, 'a') as f:
        #     print('fourth stage!!!', file=f)
        for round in range(1):
            # with open(log_file, 'a') as f:
            #     print('round: ', round, file=f)
            for i in range(max_i, -1, -1):
                if block_sum_ind_flag[i] == 1:
                    ind = block_sum_ind[i]
                    freq_m = self.square_recover(freq_m, freq_ori_m, ind)
                    freq = freq_m * np.e ** (1j * freq_p)

                    img_adv = np.abs(np.fft.ifft2(freq, axes=(0, 1)))
                    img_adv = np.clip(img_adv, 0, 1)  # 会产生一些过大值需要截断
                    # img_adv = img_adv.astype('uint8')
                    img_temp_2 = img_adv.copy()
                    # img_adv = img_adv.unsqueeze(0)
                    # img_adv = norm(img_adv)
                    # img_adv = img_adv.to(device)
                    query_num += 1
                    # with torch.no_grad():
                    #     out = net(img_adv)
                    #     _, adv_label = torch.max(out, dim=1)
                    _, adv_label = get_label_conf(self.estimator.predict(np.expand_dims(np.transpose(img_adv,(2,0,1)), axis=0).astype(MY_NUMPY_DTYPE), batch_size=self.batch_size))

                    if adv_label == ori_label:
                        freq_m = self.square_zero(freq_m, ind)
                        freq = freq_m * np.e ** (1j * freq_p)
                        # freq = square_zero(freq, ind)   # accident，这是错误的
                    else:
                        img_temp = img_temp_2.copy()
                        optimize_block += 1
                        # print('optimize block: ', i)
                        # l2_norm = torch.norm((ori_img.squeeze()) - (img_adv.squeeze()), p=2)
                        # linf_norm = torch.norm((ori_img.squeeze()) - (img_adv.squeeze()), p=np.inf)
                        # print(l2_norm.item())
                        block_sum_ind_flag[i] = 0
                
        l2_norm = np.linalg.norm(ori_img.reshape(-1) - img_adv.reshape(-1), ord=2)
        linf_norm = np.linalg.norm(ori_img.reshape(-1) - img_adv.reshape(-1), ord=np.inf)
        # print("l2_norm:{}, linf_norm:{}".format(l2_norm, linf_norm))
            # with open(log_file, 'a') as f:
            #     print('optimize block number: ', optimize_block, file=f)
            #     print('l2_norm: ', l2_norm.item(), file=f)
            #     print('linf_norm: ', linf_norm.item(), file=f)

        # with open(log_file, 'a') as f:
        #     print('final result', file=f)
        #     print('original_que number: ', query_num, file=f)
        #     print('optimize block number: ', optimize_block, file=f)
        #     print('zero block number: ', np.sum(block_sum_ind_flag), file=f)


        # img_temp = Image.fromarray(img_temp)
        # img_temp.save(save_path)
        # with open(log_file, 'a') as f:
        #     print('query number: ', query_num, file=f)
        #     print('l2_norm: ', l2_norm.item(), file=f)
        #     print('linf_norm: ', linf_norm.item(), file=f)
        return np.transpose(img_adv, (2,0,1)).astype(MY_NUMPY_DTYPE)
    
    # 输入的x已经可以直接传入神经网络进行预测
    # 需要先将x转化为图片
    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        sample_num = len(x)
        y = np.zeros(shape=x.shape)
        for i in tqdm(range(sample_num), desc="Fastdrop"):
            y[i] = self.generate_by_one_img(x[i])
        return y.astype(MY_NUMPY_DTYPE)