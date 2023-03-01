import numpy as np
from typing import List, Optional
from typing import Union
from random import randint, uniform
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from control.defense.utils.generate_aes import generate_adv_examples
from control.defense.utils.util import UnNorm

class Pixel_Deflection(object):
    def __init__(self, model:Module, 
                mean:List[float]=[0.485, 0.456, 0.406], 
                std:List[float]=[0.229, 0.224, 0.225],
                denoiser_name:str = 'TVM',
                adv_examples=None,
                adv_method: Optional[str] = 'FGSM',
                adv_dataset: Optional[str] = 'CIFAR10',
                adv_nums: Optional[int] = 10000,
                device:Union[str, torch.device]='cuda',
                ):

        super(Pixel_Deflection, self).__init__()

        self.model = model
        self.norm = transforms.Normalize(mean, std)
        self.denoiser_name = denoiser_name
        if adv_examples is None and (adv_method is None or adv_nums is None):
            raise Exception('Attack method and adversarial example nums need to be specified when the adversarial sample is not specified!')
        self.adv_examples = adv_examples
        self.adv_method = adv_method
        self.adv_dataset = adv_dataset
        self.adv_nums = adv_nums
        self.device = device
        self.un_norm = UnNorm(mean, std)
        self.total_num = 0
        self.detect_num = 0
        self.detect_rate = 0


    def generate_adv_examples(self):
        return generate_adv_examples(self.model, self.adv_method, self.adv_dataset, self.adv_nums, self.device)


    def load_adv_examples(self):
        data = torch.load(self.adv_examples)
        adv_dst = TensorDataset(data["x"].float().cpu(), data["y"].long().cpu())
        adv_loader = DataLoader(
            adv_dst,
            batch_size=64,
            shuffle=False,
            num_workers=2
        )
        return adv_loader
    
    def denoiser(self, denoiser_name, img, sigma):
        from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, denoise_nl_means, wiener)
        if denoiser_name == 'wavelet':
            return denoise_wavelet(img,sigma=sigma, mode='soft', multichannel=True,convert2ycbcr=True, method='BayesShrink')
        elif denoiser_name == 'TVM':
            return denoise_tv_chambolle(img, multichannel=True)
        elif denoiser_name == 'bilateral':
            return denoise_bilateral(img, bins=1000, multichannel=True)
        elif denoiser_name == 'deconv':
            return wiener(img)
        elif denoiser_name == 'NLM':
            return denoise_nl_means(img, multichannel=True)
        else:
            raise Exception('Incorrect denoiser mentioned. Options: wavelet, TVM, bilateral, deconv, NLM')

    def pixel_deflection(self, img, rcam_prob, deflections, window):
        C, H, W = img.shape
        while deflections > 0:
            for c in range(C):
                x,y = randint(0,H-1), randint(0,W-1)

                if uniform(0,1) < rcam_prob[x,y]:
                    continue

                while True: #this is to ensure that PD pixel lies inside the image
                    a,b = randint(-1*window,window), randint(-1*window,window)
                    if x+a < H and x+a > 0 and y+b < W and y+b > 0: break
                img[c,x,y] = img[c,x+a,y+b]
                deflections -= 1
        return img

    # unnormalized image: [3,224,224]
    def pd(self, image: Tensor):
        if len(image.size()) == 2:
            image = image.unsqueeze(0)
        image = image.clone().cpu().numpy()
        map = np.zeros((image.shape[1], image.shape[2]))
        img = self.pixel_deflection(image, map, 200, 10)
        img = img.transpose((1,2,0))
        img = self.denoiser(self.denoiser_name, img, 0.04)
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img)
        return img

    def detect(self):
        if self.adv_examples is None:
            adv_imgs, clean_examples, true_labels = self.generate_adv_examples()
        else:
            adv_imgs, adv_labels = self.load_adv_examples()

        for i in range(len(adv_imgs)):
            
            test_img = adv_imgs[i]
            cln_img = clean_examples[i]
            with torch.no_grad():
                out = self.model(test_img.unsqueeze(0).to(self.device))
                _, label1 = torch.max(out, dim=1)
            '''
                out = self.model(cln_img.unsqueeze(0))
                _, label2 = torch.max(out, dim=1)
            if label1 == label2:
                self.detect_num += 1
                '''
            test_img = self.pd(self.un_norm(test_img.squeeze()))
            test_img = self.norm(test_img).to(self.device)
            with torch.no_grad():
                out = self.model(test_img.unsqueeze(0))
                _, label_fix = torch.max(out, dim=1)
                out = self.model(cln_img.unsqueeze(0))
                _, label = torch.max(out, dim=1)
            if  label != label1:
                self.total_num += 1
            if label_fix != label1 and label != label1:
                self.detect_num += 1
                # print(self.detect_num)
                
        self.detect_rate = self.detect_num/float(self.total_num)
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate
                
    def print_res(self):
        print('detect rate: ', self.detect_rate)
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate