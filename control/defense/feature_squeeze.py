from typing import List, Optional
from typing import Union
from scipy import ndimage
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from control.defense.utils.generate_aes import generate_adv_examples
from control.defense.utils import UnNorm

class feature_squeeze(object):
    def __init__(self, model:Module, 
                mean:List[float]=[0.485, 0.456, 0.406], 
                std:List[float]=[0.229, 0.224, 0.225],
                denoiser_name:str = 'depth_reduction',
                denoiser_para:int = 2,
                adv_examples=None,
                adv_method: Optional[str] = 'FGSM',
                adv_dataset: Optional[str] = 'CIFAR10',
                adv_nums: Optional[int] = 10000,
                device:Union[str, torch.device]='cuda',
                ):

        super(feature_squeeze, self).__init__()

        self.model = model
        self.norm = transforms.Normalize(mean, std)
        self.denoiser_name = denoiser_name
        self.denoiser_para = denoiser_para
        if adv_examples is None and (adv_method is None or adv_nums is None):
            raise Exception('Attack method and adversarial example nums need to be specified when the adversarial sample is not specified!')
        self.adv_examples = adv_examples
        self.adv_method = adv_method
        self.adv_dataset = adv_dataset
        self.adv_nums = adv_nums
        self.device = device
        self.un_norm = UnNorm(mean, std)
        self.thre_score = self.init_scores()
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


    # 假设传入的是两张没有norm的[0-1]内的图片
    def depth_reduction(self, img_adv, cln_img, depth):
        coef = (2 ** depth) - 1
        img1 = img_adv.clone().unsqueeze(0)
        cln_img = cln_img.clone().unsqueeze(0)
        img2 = (coef * img_adv) / 255.0
        img2 = img2.unsqueeze(0)
        img1 = self.norm(img1)
        img2 = self.norm(img2)
        cln_img = self.norm(cln_img)
        if len(img1.size()) != 4:
            img1, img2, cln_img = img1.unsqueeze(0), img2.unsqueeze(0), cln_img.unsqueeze(0)
        score = 0

        with torch.no_grad():
            out1 = self.model(img1.to(self.device))
            out2 = self.model(img2.to(self.device))
            out_cln = self.model(cln_img.to(self.device))
            _, label1 = torch.max(out_cln, dim=1)
            _, label2 = torch.max(out2, dim=1)
            if label1 == label2:
                self.detect_num += 1
            out1 = F.softmax(out1, dim=1)
            out2 = F.softmax(out2, dim=1)
            score = torch.norm(out1-out2, p=1).item()

        return score

    def median_filter_np(self, x, width, height=-1):
        """
        Median smoothing by Scipy.
        :param x: a tensor of image(s)
        :param width: the width of the sliding window (number of pixels)
        :param height: the height of the window. The same as width by default.
        :return: a modified tensor with the same shape as x.
        """
        if height == -1:
            height = width
        x = x.cpu().numpy()
        return ndimage.filters.median_filter(x, size=(1,1,width,height), mode='reflect')


    # 假设传入的是两张没有norm的[0-1]内的图片
    def median_smoothing(self, img_adv, w_size):
        img1, img2 = img_adv.clone().squeeze().unsqueeze(0), img_adv.clone().squeeze().unsqueeze(0)
        if len(img1.size()) != 4:
            img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
        img2 = self.median_filter_np(img2, w_size)
        img2 = torch.from_numpy(img2).to(self.device)
        img1 = self.norm(img1)
        img2 = self.norm(img2)

        score = 0
        with torch.no_grad():
            out1 = F.softmax(self.model(img1.to(self.device)), dim=1)
            out2 = F.softmax(self.model(img2.to(self.device)), dim=1)
            score = torch.norm(out1-out2, p=1).item()

        return score

    def init_scores(self):
        if self.denoiser_name == 'depth_reduction':
            if self.adv_dataset == 'MNIST':
                if self.denoiser_para == 1:
                    bd_thres = 0.000102
                elif self.denoiser_para == 2:
                    bd_thres = 0.0001005
                else:
                    raise Exception('Wrong choice!')
            elif self.adv_dataset == 'CIFAR10':
                if self.denoiser_para == 1:
                    bd_thres = 0.0998
                elif self.denoiser_para == 2:
                    bd_thres = 0.117
                elif self.denoiser_para == 3:
                    bd_thres = 0.1135
                elif self.denoiser_para == 4:
                    bd_thres = 0.1042
                elif self.denoiser_para == 5:
                    bd_thres = 0.09372
                else:
                    raise Exception('Wrong choice!')
            elif self.adv_dataset == 'CIFAR100':
                if self.denoiser_para == 1:
                    bd_thres = 1.9997
                elif self.denoiser_para == 2:
                    bd_thres = 1.9967
                elif self.denoiser_para == 3:
                    bd_thres = 1.7822
                elif self.denoiser_para == 4:
                    bd_thres = 0.7930
                elif self.denoiser_para == 5:
                    bd_thres = 0.3301
                else:
                    raise Exception('Wrong choice!')
            elif self.adv_dataset == 'Imagenet':
                if self.denoiser_para == 1:
                    bd_thres = 1.9942
                elif self.denoiser_para == 2:
                    bd_thres = 1.9512
                elif self.denoiser_para == 3:
                    bd_thres = 1.4417
                elif self.denoiser_para == 4:
                    bd_thres = 0.7996
                elif self.denoiser_para == 5:
                    bd_thres = 0.3528
                else:
                    raise Exception('Wrong choice!')
            else:
                raise Exception('Wrong choice!')
            return bd_thres
        elif self.denoiser_name == 'median_smoothing':
            if self.adv_dataset == 'MNIST':
                if self.denoiser_para == 2:
                    mf_thres = 0.0029
                elif self.denoiser_para == 3:
                    mf_thres = 0.039
                else:
                    raise Exception('Wrong choice!')
            elif self.adv_dataset == 'CIFAR10':
                if self.denoiser_para == 2:
                    mf_thres = 0.0647
                elif self.denoiser_para == 3:
                    mf_thres = 0.0704
                else:
                    raise Exception('Wrong choice!')
            elif self.adv_dataset == 'CIFAR100':
                if self.denoiser_para == 2:
                    mf_thres = 1.1296
                elif self.denoiser_para == 3:
                    mf_thres = 1.9431
                else:
                    raise Exception('Wrong choice!')
            elif self.adv_dataset == 'Imagenet':
                if self.denoiser_para == 2:
                    mf_thres = 1.1472
                elif self.denoiser_para == 3:
                    mf_thres = 1.6615
                else:
                    raise Exception('Wrong choice!')
            return mf_thres
        else:
            raise Exception('Input right detect method')
        return

    def fs_detect(self, adv_img, cln_img):
        if self.denoiser_name == 'depth_reduction':
            scores = self.depth_reduction(adv_img, cln_img, self.denoiser_para)
            return scores
        elif self.denoiser_name == 'median_smoothing':
            return self.median_smoothing(adv_img, self.denoiser_para)
        else:
            raise Exception('Input right detect method')

    def detect(self):
        if self.adv_examples is None:
            adv_imgs, cln_imgs, true_labels = self.generate_adv_examples()
        else:
            adv_imgs, adv_labels = self.load_adv_examples()

        for i in range(len(adv_imgs)):
            self.total_num += 1
            test_img = adv_imgs[i]
            cln_img = cln_imgs[i]
            detect_score = self.fs_detect(self.un_norm(test_img).squeeze(), self.un_norm(cln_img).squeeze())
            '''
            if detect_score >= self.thre_score:
                self.detect_num += 1
                print(self.detect_num)
            '''
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