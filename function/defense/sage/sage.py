import os
import torch
from torch import nn
from torch.nn import Module
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional
from typing import Union
from function.defense.sage.at_SAD import AT
from function.defense.sage.utils.util import *
from function.defense.sage.resnet_tmp import *
from function.defense.utils.generate_aes import generate_adv_examples
def train_step(cuda, train_loader, nets, optimizer, criterions, epoch, beta1, beta2, beta3):
    at_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets#['snet'] #

    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']

    snet.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        if cuda:
            img = img.cuda()
            target = target.cuda()

        activation1_s, activation2_s, activation3_s, activation4_s, output_s = snet(img)
        cls_loss = criterionCls(output_s, target.long())

        at_loss1 = criterionAT(activation1_s, activation2_s) * beta1
        at_loss2 = criterionAT(activation2_s, activation3_s) * beta2
        at_loss3 = criterionAT(activation3_s, activation4_s) * beta3
        at_loss = at_loss1 + cls_loss + at_loss2 + at_loss3

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        at_losses.update(at_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        at_loss.backward()
        optimizer.step()

        if idx % 20 == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'AT_loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=at_losses,
                                                                 top1=top1, top5=top5))


def test(test_clean_loader, test_bad_loader, model, criterions, epoch, beta1, beta2, beta3):
    top1 = AverageMeter()
    top5 = AverageMeter()
    atcls_losses = AverageMeter()

    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']

    model.eval()

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            activation1_s, activation2_s, activation3_s, activation4_s, output_s = model(img)

            at_loss1 = criterionAT(activation1_s, activation2_s).detach() * beta1
            at_loss2 = criterionAT(activation2_s, activation3_s).detach() * beta2
            at_loss3 = criterionAT(activation3_s, activation4_s).detach() * beta3
            at_loss = at_loss1 + at_loss2 + at_loss3
            cls_loss = criterionCls(output_s, target.long())
            atcls_loss = at_loss + cls_loss

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))
        atcls_losses.update(atcls_loss.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg, atcls_losses.avg]

    cls_losses = AverageMeter()
    at_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        img = img.cuda()
        target = target.cuda()
        with torch.no_grad():
            activation1_s, activation2_s, activation3_s, activation4_s, output_s = model(img)

            at_loss1 = criterionAT(activation1_s, activation3_s).detach() * beta1
            at_loss2 = criterionAT(activation2_s, activation4_s).detach() * beta2
            at_loss3 = criterionAT(activation3_s, activation4_s).detach() * beta3
            at_loss = at_loss1 + at_loss2 + at_loss3
            cls_loss = criterionCls(output_s, target.long())

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        at_losses.update(at_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, cls_losses.avg, at_losses.avg]
    curr_loss = acc_clean[2]

    print('[epoch]:', epoch)
    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

    return acc_clean, acc_bd, curr_loss


'''
    :param model: nn.Module
    :param dataset: string
    :param test_clean_loader: DataLoader
    :param test_bad_loader: DataLoader
    :param train_clean_loader: DataLoader
    :param cuda: int, 1 or 0
'''
def sage(model, dataset, test_clean_loader, test_bad_loader, train_clean_loader, cuda, 
         p = 2, beta1 = 300, beta2 = 300, beta3 = 300, epochs = 100):
    if(dataset == 'cifar10'):
        lr = 0.05
    elif(dataset == 'cifar100'):
        lr = 0.08
    elif(dataset == 'mnist'):
        lr = 0.001
    # initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=1e-4,
                                nesterov=True)
    # define loss functions
    if cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
        criterionAT = AT(p)
    else:
        criterionCls = nn.CrossEntropyLoss()
        criterionAT = AT(p)
    best_acc_bad = 0 #
    for epoch in range(0, epochs):
        # train every epoch
        criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}

        if epoch == 0:
            # before training test firstly
            acc_clean_before, acc_bad_before, curr_loss = test(test_clean_loader, test_bad_loader, model, criterions, epoch, beta1, beta2, beta3)
        train_step(cuda, train_clean_loader, model, optimizer, criterions, epoch + 1, beta1, beta2, beta3)
        # evaluate on testing set
        print('testing the models......')
        
        acc_clean, acc_bad, curr_loss = test(test_clean_loader, test_bad_loader, model, criterions, epoch + 1, beta1, beta2, beta3)
        if epoch == 1:
            best_acc_clean = acc_clean[0]
            best_acc_bad = acc_bad[0]
        if epoch == 0:
            pass
        else:
            if acc_bad[0] < best_acc_bad:
                best_acc_clean = acc_clean[0]
                best_acc_bad = acc_bad[0]
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_clean_acc': best_acc_clean,
                    'best_bad_acc': best_acc_bad,
                    'optimizer': optimizer.state_dict(),
                }, './weight/results', 'res')
    return best_acc_clean, best_acc_bad, acc_clean_before[0], acc_bad_before[0]

class Sage(object):
    def __init__(self, model:Module, 
                mean:List[float]=[0.485, 0.456, 0.406], 
                std:List[float]=[0.229, 0.224, 0.225],
                adv_examples=None,
                adv_method: Optional[str] = 'PGD',
                adv_dataset: Optional[str] = 'CIFAR10',
                adv_nums: Optional[int] = 10,#
                device:Union[str, torch.device]='cuda',
                ):

        super(Sage, self).__init__()

        self.model = model
        self.norm = transforms.Normalize(mean, std)
        if adv_examples is None and (adv_method is None or adv_nums is None):
            raise Exception('Attack method and adversarial example nums need to be specified when the adversarial sample is not specified!')
        self.adv_examples = adv_examples
        self.adv_method = adv_method
        self.adv_dataset = adv_dataset
        self.adv_nums = adv_nums
        self.device = device
        # self.un_norm = UnNorm(mean, std)
        self.total_num = 0
        self.detect_num = 0
        self.detect_rate = 0

    def generate_adv_examples(self):
        return generate_adv_examples(self.model, self.adv_method, self.adv_dataset, self.adv_nums, self.device)

    def load_adv_examples(self):
        data = torch.load(self.adv_examples)
        print('successfully load adversarial examples!')
        return data['adv_img'], data['cln_img'], data['y']

    def detect(self):
        if self.adv_examples is None:
            adv_imgs, cln_imgs, true_labels = self.generate_adv_examples()
        else:
            adv_imgs, cln_imgs, true_labels = self.load_adv_examples() 

        """
        Load model
        """
        model = resnet18(10)
        backdoor_path = "/mnt/data2/yxl/AI-platform/model/robnet-resnet18-cifar10.pth"
        model_path = os.path.join(backdoor_path)
        print('Loading Model from {}'.format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_dict'])
        model = model.to('cuda')
        
        """Load test clean dataset and bad dataset"""
        tf_test = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010],
                                    )])
        testset = datasets.CIFAR10(
                root='/mnt/data2/yxl/AI-platform/dataset/CIFAR10', train=False, download=True)
        test_data = torch.load('/mnt/data2/yxl/AI-platform/dataset/CIFAR10/cifar10_poisoned_samples_test.pth')
        
        test_data_clean = DatasetFull(
            full_dataset=testset, transform=tf_test)
        
        test_clean_loader = DataLoader(dataset=test_data_clean,
                                    batch_size=64,
                                    shuffle=False,
                                    )
        # all clean test data
        """Load clean train dataset"""
        test_bad_loader = DataLoader(dataset=test_data,
                                    batch_size=64,
                                    shuffle=False,)
        
        tf_train = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                    mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010],
                                    )])
        trainset = datasets.CIFAR10(
                root='/mnt/data2/yxl/AI-platform/dataset/CIFAR10', train=True, download=True)
        
        train_data = DatasetPart(ratio = 0.1, full_dataset=trainset, transform=tf_train)
        train_loader = DataLoader(
            train_data, batch_size=64, shuffle=True)
        
        
        results = sage(model, 'cifar10', test_clean_loader, test_bad_loader, train_loader, cuda = 1, epochs=2)
        detect_rate = (100 - results[1]) / 100
        self.detect_rate = float(detect_rate)
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate
    
    def print_res(self):
        print('detect rate: ', self.detect_rate)