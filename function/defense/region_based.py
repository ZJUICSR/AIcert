from typing import List, Optional
from typing import Union
import torch
import torchvision.transforms as transforms
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
from function.defense.utils.generate_aes import generate_adv_examples
from function.defense.utils.util import UnNorm

class RegionBased(object):
    def __init__(self, model:Module, 
                mean:List[float]=[0.485, 0.456, 0.406], 
                std:List[float]=[0.229, 0.224, 0.225],
                radius:float=0.1,
                sample_num:int=1000,
                adv_examples=None,
                adv_method: Optional[str] = 'PGD',
                adv_dataset: Optional[str] = 'CIFAR10',
                adv_nums: Optional[int] = 10000,
                device:Union[str, torch.device]='cuda',
                ):

        super(RegionBased, self).__init__()

        self.model = model
        self.norm = transforms.Normalize(mean, std)
        if adv_examples is None and (adv_method is None or adv_nums is None):
            raise Exception('Attack method and adversarial example nums need to be specified when the adversarial sample is not specified!')
        self.radius = radius
        self.sample_num = sample_num
        self.adv_examples = adv_examples
        self.adv_method = adv_method
        self.adv_dataset = adv_dataset
        self.adv_nums = adv_nums
        self.device = device
        self.total_num = 0
        self.detect_num = 0
        self.detect_rate = 0
        self.un_norm = UnNorm(mean, std)


    def generate_adv_examples(self):
        return generate_adv_examples(self.model, self.adv_method, self.adv_dataset, self.adv_nums, self.device)


    def load_adv_examples(self):
        data = torch.load(self.adv_examples)
        adv_dst = TensorDataset(data["x"].float().cpu(), data["y"].long().cpu())
        adv_loader = DataLoader(
        adv_dst,
        batch_size=1,
        shuffle=False,
        num_workers=2
        )
        return adv_loader


    def region_based(self, img):
        label_count = {}
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # if torch.cuda.is_available():
        #     print("got GPU")
        for i in range(self.sample_num):
            if self.adv_dataset == 'CIFAR10':
                noise = torch.rand((3,32,32)).to(self.device)
            elif self.adv_dataset == 'MNIST':
                noise = torch.rand((1,28,28)).to(self.device)
            noise = noise*2.0 - 1.0
            noise = noise * self.radius
            img_noise = img.squeeze() + noise
            img_noise = torch.clamp(img_noise, 0.0, 1.0)
            img_noise = self.norm(img_noise)
            with torch.no_grad():
                out = self.model(img_noise.unsqueeze(0).to('cuda:0'))
                _, label_noise = torch.max(out, dim=1)
            if label_noise not in label_count.keys():
                label_count[label_noise] = 1
            else:
                label_count[label_noise] += 1
        adv_label = None
        max_num = -1
        for k in label_count.keys():
            if label_count[k] > max_num:
                adv_label = k
                max_num = label_count[k]

        return adv_label


    def region_based_tmp(self, img):
        label_count = {}

        if self.sample_num < 100:
            loop_times = 1
            batch_size = self.sample_num
        else:
            loop_times = self.sample_num // 100
            batch_size = 100

        for i in range(loop_times):
            if self.adv_dataset == 'CIFAR10':
                noise = torch.rand((batch_size,3,32,32)).to(self.device)
            elif self.adv_dataset == 'MNIST':
                noise = torch.rand((batch_size,1,28,28)).to(self.device)
            imgs = img.squeeze().repeat(batch_size,1,1,1)    
            noise = noise*2.0 - 1.0
            noise = noise * self.radius
            img_noise = imgs + noise
            img_noise = torch.clamp(img_noise, 0.0, 1.0)
            img_noise = self.norm(img_noise)
            with torch.no_grad():
                out = self.model(img_noise.to('cuda:0'))
                _, labels_noise = torch.max(out, dim=1)
            for label_noise in labels_noise:
                if label_noise not in label_count.keys():
                    label_count[label_noise] = 1
                else:
                    label_count[label_noise] += 1
        adv_label = None
        max_num = -1
        for k in label_count.keys():
            if label_count[k] > max_num:
                adv_label = k
                max_num = label_count[k]

        return adv_label

    def detect(self):
        if self.adv_examples is None:
            adv_imgs, clean_imgs, true_labels = self.generate_adv_examples()
        else:
            adv_imgs, adv_labels = self.load_adv_examples()

        for i in range(len(adv_imgs)):
            self.total_num += 1
            test_img = adv_imgs[i]
            clean_img = clean_imgs[i]
            with torch.no_grad():
                out = self.model(test_img.unsqueeze(0).to(self.device))
                _, label = torch.max(out, dim=1)
            with torch.no_grad():
                out = self.model(clean_img.unsqueeze(0).to(self.device))
                _, cln_label = torch.max(out, dim=1)
            defense_label = self.region_based_tmp(self.un_norm(test_img))
            #if defense_label != label:
            if defense_label == cln_label[0]:
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

