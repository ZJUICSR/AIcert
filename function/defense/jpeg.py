import numpy as np
from typing import List, Optional
from typing import Union
import torch
from PIL import Image
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from function.defense.utils.generate_aes import generate_adv_examples
from function.defense.DiffJPEG.DiffJPEG import DiffJPEG
from function.defense.utils.util import UnNorm

class Jpeg(object):
    def __init__(self, model:Module, 
                mean:List[float]=[0.485, 0.456, 0.406], 
                std:List[float]=[0.229, 0.224, 0.225],
                jpeg_quality:float = 80,
                adv_examples=None,
                adv_method: Optional[str] = 'PGD',
                adv_dataset: Optional[str] = 'CIFAR10',
                adv_nums: Optional[int] = 1000,
                device:Union[str, torch.device]='cuda',
                ):

        super(Jpeg, self).__init__()

        self.model = model
        self.norm = transforms.Normalize(mean, std)
        self.jpeg_quality = jpeg_quality
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
        batch_size=1,
        shuffle=False,
        num_workers=2
        )
        return adv_loader

    # unnormalized image: [3,224,224]
    def jpeg(self, image: Tensor):
        image = image.clone().cpu().numpy()
        C, H, W = image.shape
        jpeg_func = DiffJPEG(height=H, width=W, differentiable=True, quality=self.jpeg_quality)
        img = torch.from_numpy(image)
        img = jpeg_func(img.unsqueeze(0))
        return img

    def jpeg_compression(self, img:torch.Tensor, quality:int=75, save_path:str='./temp.JPEG'):
        img1 = img.cpu().squeeze().numpy().copy()
        img1 = img1 * 255.0
        img1 = np.clip(img1, 0, 255)
        img1 = img1.astype('uint8')
        if len(img1.shape) == 3:
            img1 = img1.transpose(1,2,0)
        img1 = Image.fromarray(img1)
        img1.save(save_path, "JPEG", quality=quality)
        img1 = Image.open(save_path)
        trans1 = transforms.ToTensor()
        img1 = trans1(img1)

        return img1


    def detect(self):
        if self.adv_examples is None:
            adv_imgs, clean_examples, true_labels = self.generate_adv_examples()
        else:
            adv_imgs, adv_labels = self.load_adv_examples()

        for i in range(len(adv_imgs)):
            self.total_num += 1
            test_img = adv_imgs[i]
            with torch.no_grad():
                out = self.model(test_img.unsqueeze(0).to(self.device))
                _, label = torch.max(out, dim=1)
            test_img = self.jpeg(self.un_norm(test_img))
            test_img = test_img.squeeze().unsqueeze(0)
            if len(test_img.size()) != 4:
                test_img = test_img.unsqueeze(0)
            test_img = test_img.to(self.device)
            with torch.no_grad():
                out = self.model(test_img)
                _, label_fix = torch.max(out, dim=1)
            if label_fix != label:
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