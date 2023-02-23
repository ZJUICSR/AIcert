# 改造开始
from __future__ import absolute_import, division, print_function, unicode_literals
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List
import numpy as np
import torch
import torchvision.transforms as transforms
from utils.util import UnNorm
import logging
from typing import Optional, TYPE_CHECKING
import numpy as np
from scipy.fftpack import idct
from tqdm.auto import trange
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.config import ART_NUMPY_DTYPE
from art.utils import is_probability
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class Fastdrop(EvasionAttack):

    attack_params = EvasionAttack.attack_params + [
        "attack",
        "max_iter",
        "epsilon",
        "order",
        "freq_dim",
        "stride",
        "targeted",
        "batch_size",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin, NeuralNetworkMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        MEAN = [0.4914, 0.4822, 0.4465],
        STD = [0.2023, 0.1994, 0.2010],
        datatrans = None,
        redatatrans = None,
    ):
        super().__init__(estimator=classifier)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.trans1 = transforms.ToTensor()
    
        def datatrans(data) -> np.ndarray:
            data = np.transpose(data, (0, 3, 1, 2)).astype(np.float32)
            return data
        
        def redatatrans(data) -> np.ndarray:
            data = np.transpose(data, (0, 2, 3, 1)).astype(np.float32)
            return data
    
        self.datatrans = datatrans
        self.redatatrans = redatatrans
    
    def square_zero(self, freq:np.ndarray, index:int, img_shape:int):
        freq_modified = freq.copy()
        freq_modified[index, index:img_shape-index, :] = 0
        freq_modified[img_shape-1-index, index:img_shape-index, :] = 0
        freq_modified[index:img_shape-index:, index, :] = 0
        freq_modified[index:img_shape-index, img_shape-1-index, :] = 0

        return freq_modified

    def square_avg(self, freq:np.ndarray, index:int, img_shape:int):
        rank1 = np.sum(freq[index, index:img_shape-index, :])
        rank2 = np.sum(freq[img_shape-1-index, index:img_shape-index, :])
        col1 = np.sum(freq[index+1:img_shape-1-index, index, :])
        col2 = np.sum(freq[index+1:img_shape-1-index, img_shape-1-index, :])
        num = 4*(img_shape - 2*index) - 2
        return (rank1+rank2+col1+col2) / float(num)
    
    def square_recover(self, freq_modified:np.ndarray, freq_ori:np.ndarray, index:int, img_shape:int):
        freq_modified[index, index:img_shape-index, :] = freq_ori[index, index:img_shape-index, :]
        freq_modified[img_shape-1-index, index:img_shape-index, :] = freq_ori[img_shape-1-index, index:img_shape-index, :]
        freq_modified[index:img_shape-index:, index, :] = freq_ori[index:img_shape-index:, index, :]
        freq_modified[index:img_shape-index, img_shape-1-index, :] = freq_ori[index:img_shape-index, img_shape-1-index, :]

        return freq_modified

    # input为img，output为输入到神经网络的预处理之后的图片square_avg
    def generate_by_one_img(self, img:np.ndarray):
        query_num = 0
        ori_img = img.copy()

        testimg1 = self.trans1(img)
        testimg1 = testimg1.to(self.device)

        # 第一次查询给出原始标签
        query_num += 1
        ori_label = np.argmax(self.estimator.predict(self.datatrans(np.expand_dims(img, axis=0))), axis=1)

        # DFT
        # fft to original numpy image
        freq = np.fft.fft2(ori_img, axes=(0, 1))
        freq_ori = freq.copy()
        # 计算绝对值
        freq_ori_m = np.abs(freq_ori)
        freq_abs = np.abs(freq)
        # 分块
        # img_shape=32 32*32 按照2*2进行分块，分成16*16块
        num_block = int(np.shape(img)[0] / 2)
        block_sum = np.zeros(num_block)
        for i in range(num_block):
                block_sum[i] = self.square_avg(freq_abs, i, np.shape(img)[2])

        # ordered index
        block_sum_ind = np.argsort(block_sum)
        # block_sum_ind = block_sum_ind[::-1]
        block_sum_ind_flag = np.zeros(num_block)

        img_save = None
        mags = list(range(int(num_block/2), num_block+1))
        freq_sec_stage = freq.copy()
        freq_sec_stage_m = np.abs(freq_sec_stage)   # 幅度谱
        freq_sec_stage_p = np.angle(freq_sec_stage) # 相位谱
        mag_start = 0

        for mag in mags:
            for i in range(mag_start, mag):
                ind = block_sum_ind[i]
                freq_sec_stage_m = self.square_zero(freq_sec_stage_m, ind, np.shape(img)[2])
                freq_sec_stage = freq_sec_stage_m * np.e ** (1j * freq_sec_stage_p)

            img_adv = np.abs(np.fft.ifft2(freq_sec_stage, axes=(0, 1)))
            img_adv = np.clip(img_adv, 0, 255)  # 会产生一些过大值需要截断
            img_adv = img_adv.astype('uint8')
            img_save = img_adv.copy()

            testimg2 = self.trans1(img_adv)
            # testimg2 = self.norm(testimg2)
            testimg2 = testimg2.to(self.device)
            query_num += 1
            # with torch.no_grad():
            #     out = net(img_adv)
            #     _, adv_label = torch.max(out, dim=1)
            adv_label = np.argmax(self.estimator.predict(self.datatrans(np.expand_dims(img_adv, axis=0))), axis=1)

            mag_start = mag
            if ori_label != adv_label:
                print('%d block' % (mag))
                print('l2_norm: ', torch.norm((testimg1.squeeze()) - (testimg2.squeeze()), p=2).item())
                print('linf_norm: ', torch.norm((testimg1.squeeze()) - (testimg2.squeeze()), p=np.inf).item())
                l2_norm = torch.norm((testimg1.squeeze()) - (testimg2.squeeze()), p=2)
                if l2_norm.item() < 1.5:
                    # with open(log_file, 'a') as f:
                    #     print('%d block success' % (mag), file=f)
                    # img_save = Image.fromarray(img_save)
                    # img_save.save(save_path)
                    # with open(log_file, 'a') as f:
                    #     # print('binary search', file=f)
                    #     print('query number: ', query_num, file=f)
                    #     print('l2_norm: ', torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=2).item(), file=f)
                    #     print('linf_norm: ', torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=np.inf).item(), file=f)

                    return img_adv
        
        img_save = None
        img_temp = img_save
        # max_i = -1
        max_i = mag_start - 1
        block_sum_ind_flag[:max_i+1] = 1
        print('max_i: ', max_i)
        # freq_m = np.abs(freq)
        freq_m = freq_sec_stage_m
        freq_p = np.angle(freq)


        # optimize the adv example
        optimize_block = 0
        l2_norm = torch.tensor(0)
        linf_norm = torch.tensor(0)

        for round in range(2):
            for i in range(max_i, -1, -1):
                if block_sum_ind_flag[i] == 1:
                    ind = block_sum_ind[i]
                    freq_m = self.square_recover(freq_m, freq_ori_m, ind, np.shape(img)[2])
                    freq = freq_m * np.e ** (1j * freq_p)

                    img_adv = np.abs(np.fft.ifft2(freq, axes=(0, 1)))
                    img_adv = np.clip(img_adv, 0, 255)  # 会产生一些过大值需要截断
                    img_adv = img_adv.astype('uint8')
                    img_temp_2 = img_adv.copy()

                    testimg2 = self.trans1(img_adv)
                    # testimg2 = self.norm(testimg2)
                    testimg2 = testimg2.to(self.device)

                    query_num += 1
                    # with torch.no_grad():
                    #     out = net(img_adv)
                    #     _, adv_label = torch.max(out, dim=1)
                    adv_label = np.argmax(self.estimator.predict(self.datatrans(np.expand_dims(img_adv, axis=0))), axis=1)

                    if adv_label == ori_label:
                        freq_m = self.square_zero(freq_m, ind, np.shape(img)[2])
                        freq = freq_m * np.e ** (1j * freq_p)
                        # freq = square_zero(freq, ind)   # accident，这是错误的
                    else:
                        img_temp = img_temp_2.copy()
                        optimize_block += 1
                        # print('optimize block: ', i)
                        l2_norm = torch.norm((testimg1.squeeze()) - (testimg2.squeeze()), p=2)
                        linf_norm = torch.norm((testimg1.squeeze()) - (testimg2.squeeze()), p=np.inf)
                        # print(l2_norm.item())
                        block_sum_ind_flag[i] = 0
            if optimize_block == 0: # will not happen
                l2_norm = torch.norm((testimg1.squeeze()) - (testimg2.squeeze()), p=2)
                linf_norm = torch.norm((testimg1.squeeze()) - (testimg2.squeeze()), p=np.inf)
        return img_temp
    
    # 输入的x已经可以直接传入神经网络进行预测
    # 需要先将x转化为图片
    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        # 转化x成为图片
        tmpx = x.copy()
        img = self.redatatrans(tmpx).copy()
        # 图片总数
        num = len(img)
        # 初始化产出
        y = np.zeros(shape=img.shape)
        for i in tqdm(range(num), desc="正在生成"):
            y[i] = self.generate_by_one_img(img[i])
        return self.datatrans(y)