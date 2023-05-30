import os
import torchvision
import torch.optim as optim
from torchvision import transforms
from models import *
from tqdm import tqdm
import numpy as np
import copy

from utils import Logger, save_checkpoint, torch_accuracy, AverageMeter
from attacks import *

train_loader = 0
test_loader = 0
adjust_learning_rate = 0


def seat(args_dict):
    # Training settings
    args_dict['out_dir'] = os.path.join(args_dict['out_dir'], args_dict['ablation'])
    if not os.path.exists(args_dict['out_dir']):
        os.makedirs(args_dict['out_dir'])

    args_dict['num_classes'] = 10
    weight_decay = 3.5e-3 if args_dict['arch'] == 'resnet18' else 7e-4
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='/data/user/WZT/Datasets/SEAT_data/train', train=True, download=True,
                                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='/data/user/WZT/Datasets/SEAT_data/test', train=False, download=True,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    if args_dict['arch'] == 'resnet18':
        adjust_learning_rate = lambda epoch: \
            np.interp([epoch], [0, args_dict['epochs'] // 3, args_dict['epochs'] * 2 // 3, args_dict['epochs']],
                      [args_dict['lr'], args_dict['lr'], args_dict['lr'] / 10, args_dict['lr'] / 100])[0]
    elif args_dict['arch'] == 'WRN':
        args_dict['lr'] = 0.1
        adjust_learning_rate = lambda epoch: \
            np.interp([epoch], [0, args_dict['epochs'] // 3, args_dict['epochs'] * 2 // 3, args_dict['epochs']],
                      [args_dict['lr'], args_dict['lr'], args_dict['lr'] / 10, args_dict['lr'] / 20])[0]
    best_acc_clean = 0
    best_acc_adv = best_ema_acc_adv = 0
    start_epoch = 1

    if args_dict['arch'] == "smallcnn":
        model = SmallCNN()
    if args_dict['arch'] == "resnet18":
        model = ResNet18(num_classes=args_dict['num_classes'])
    if args_dict['arch'] == "preactresnet18":
        model = PreActResNet18(num_classes=args_dict['num_classes'])
    if args_dict['arch'] == "WRN":
        model = Wide_ResNet_Madry(depth=32, num_classes=args_dict['num_classes'], widen_factor=10, dropRate=0.0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = torch.device("cuda")
    model = torch.nn.DataParallel(model)
    teacher_model = EMA(model)
    model = model.to(device)
    Attackers = AttackerPolymer(args_dict['epsilon'], args_dict['num_steps'], args_dict['step_size'],
                                args_dict['num_classes'], device)

    if not args_dict['resume']:
        optimizer = optim.SGD(model.parameters(), lr=args_dict['lr'], momentum=0.9, weight_decay=weight_decay)

        logger_test = Logger(os.path.join(args_dict['out_dir'], 'log_results.txt'), title='reweight')
        logger_test.set_names(['Epoch', 'Natural', 'PGD20', 'ema_Natural', 'ema_PGD20'])

        for epoch in range(start_epoch, args_dict['epochs'] + 1):

            descrip_str = 'Training epoch:{}/{}'.format(epoch, args_dict['epochs'])

            train(epoch, model, teacher_model, Attackers, optimizer, device, descrip_str)
            nat_acc, pgd20_acc, ema_nat_acc, ema_pgd20_acc = test(model, teacher_model, Attackers, device=device)

            logger_test.append([epoch, nat_acc, pgd20_acc, ema_nat_acc, ema_pgd20_acc])

            if pgd20_acc > best_acc_adv:
                print('==> Updating the best model..')
                best_acc_adv = pgd20_acc
                torch.save(model.state_dict(), os.path.join(args_dict['out_dir'], 'bestpoint.pth.tar'))

            if ema_pgd20_acc > best_ema_acc_adv:
                print('==> Updating the teacher model..')
                best_ema_acc_adv = ema_pgd20_acc
                torch.save(teacher_model.model.state_dict(),
                           os.path.join(args_dict['out_dir'], 'ema_bestpoint.pth.tar'))

            # Save the last checkpoint
            torch.save(model.state_dict(), os.path.join(args_dict['out_dir'], 'lastpoint.pth.tar'))

    model.load_state_dict(torch.load(os.path.join(args_dict['out_dir'], 'bestpoint.pth.tar')))
    teacher_model.model.load_state_dict(torch.load(os.path.join(args_dict['out_dir'], 'ema_bestpoint.pth.tar')))
    res_list = attack(model, Attackers, device)
    res_list1 = attack(teacher_model.model, Attackers, device)

    logger_test.set_names(
        ['Epoch', 'clean', 'PGD20', 'PGD100', 'MIM', 'CW', 'APGD_ce', 'APGD_dlr', 'APGD_t', 'FAB_t', 'Square', 'AA'])
    logger_test.append(
        [1000000, res_list[0], res_list[1], res_list[2], res_list[3], res_list[4], res_list[5], res_list[6],
         res_list[7], res_list[8], res_list[9], res_list[10]])
    logger_test.append(
        [1000001, res_list1[0], res_list1[1], res_list1[2], res_list1[3], res_list1[4], res_list1[5], res_list1[6],
         res_list1[7], res_list1[8], res_list1[9], res_list1[10]])
    logger_test.close()
    return nat_acc, pgd20_acc, ema_nat_acc, ema_pgd20_acc


class EMA(object):
    def __init__(self, model, alpha=0.999, buffer_ema=True):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self, model):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }


def train(epoch, model, teacher_model, Attackers, optimizer, device, descrip_str):
    teacher_model.model.eval()

    losses = AverageMeter()
    clean_accuracy = AverageMeter()
    adv_accuracy = AverageMeter()

    pbar = tqdm(train_loader)
    pbar.set_description(descrip_str)

    for batch_idx, (inputs, target) in enumerate(pbar):
        pbar_dic = OrderedDict()

        inputs, target = inputs.to(device), target.to(device)

        x_adv = Attackers.run_specified('PGD_10', model, inputs, target, return_acc=False)

        model.train()
        lr = adjust_learning_rate(epoch)
        optimizer.param_groups[0].update(lr=lr)
        optimizer.zero_grad()

        nat_logit = teacher_model.model(inputs)

        logit = model(x_adv)
        loss = nn.CrossEntropyLoss()(logit, target)
        loss.backward()
        optimizer.step()

        teacher_model.update_params(model)
        teacher_model.apply_shadow()

        losses.update(loss.item())
        clean_accuracy.update(torch_accuracy(nat_logit, target, (1,))[0].item())
        adv_accuracy.update(torch_accuracy(logit, target, (1,))[0].item())

        pbar_dic['loss'] = '{:.2f}'.format(losses.mean)
        pbar_dic['Acc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['advAcc'] = '{:.2f}'.format(adv_accuracy.mean)
        pbar.set_postfix(pbar_dic)


def test(model, teacher_model, Attackers, device):
    model.eval()
    teacher_model.model.eval()

    clean_accuracy = AverageMeter()
    adv_accuracy = AverageMeter()
    ema_clean_accuracy = AverageMeter()
    ema_adv_accuracy = AverageMeter()

    pbar = tqdm(test_loader)
    pbar.set_description('Testing')

    for batch_idx, (inputs, target) in enumerate(pbar):
        pbar_dic = OrderedDict()

        inputs, target = inputs.to(device), target.to(device)

        acc = Attackers.run_specified('NAT', model, inputs, target, return_acc=True)
        adv_acc = Attackers.run_specified('PGD_20', model, inputs, target, category='Madry', return_acc=True)

        ema_acc = Attackers.run_specified('NAT', teacher_model.model, inputs, target, return_acc=True)
        ema_adv_acc = Attackers.run_specified('PGD_20', teacher_model.model, inputs, target, category='Madry',
                                              return_acc=True)

        clean_accuracy.update(acc[0].item(), inputs.size(0))
        adv_accuracy.update(adv_acc[0].item(), inputs.size(0))
        ema_clean_accuracy.update(ema_acc[0].item(), inputs.size(0))
        ema_adv_accuracy.update(ema_adv_acc[0].item(), inputs.size(0))

        pbar_dic['cleanAcc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['advAcc'] = '{:.2f}'.format(adv_accuracy.mean)
        pbar_dic['ema_cleanAcc'] = '{:.2f}'.format(ema_clean_accuracy.mean)
        pbar_dic['ema_advAcc'] = '{:.2f}'.format(ema_adv_accuracy.mean)
        pbar.set_postfix(pbar_dic)

    return clean_accuracy.mean, adv_accuracy.mean, ema_clean_accuracy.mean, ema_adv_accuracy.mean


def attack(model, Attackers, device):
    model.eval()

    clean_accuracy = AverageMeter()
    pgd20_accuracy = AverageMeter()
    pgd100_accuracy = AverageMeter()
    mim_accuracy = AverageMeter()
    cw_accuracy = AverageMeter()
    APGD_ce_accuracy = AverageMeter()
    APGD_dlr_accuracy = AverageMeter()
    APGD_t_accuracy = AverageMeter()
    FAB_t_accuracy = AverageMeter()
    Square_accuracy = AverageMeter()
    aa_accuracy = AverageMeter()

    pbar = tqdm(test_loader)
    pbar.set_description('Attacking all')

    for batch_idx, (inputs, targets) in enumerate(pbar):
        pbar_dic = OrderedDict()

        inputs, targets = inputs.to(device), targets.to(device)

        acc_dict = Attackers.run_all(model, inputs, targets)

        clean_accuracy.update(acc_dict['NAT'][0].item(), inputs.size(0))
        pgd20_accuracy.update(acc_dict['PGD_20'][0].item(), inputs.size(0))
        pgd100_accuracy.update(acc_dict['PGD_100'][0].item(), inputs.size(0))
        mim_accuracy.update(acc_dict['MIM'][0].item(), inputs.size(0))
        cw_accuracy.update(acc_dict['CW'][0].item(), inputs.size(0))
        APGD_ce_accuracy.update(acc_dict['APGD_ce'][0].item(), inputs.size(0))
        APGD_dlr_accuracy.update(acc_dict['APGD_dlr'][0].item(), inputs.size(0))
        APGD_t_accuracy.update(acc_dict['APGD_t'][0].item(), inputs.size(0))
        FAB_t_accuracy.update(acc_dict['FAB_t'][0].item(), inputs.size(0))
        Square_accuracy.update(acc_dict['Square'][0].item(), inputs.size(0))
        aa_accuracy.update(acc_dict['AA'][0].item(), inputs.size(0))

        pbar_dic['clean'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['PGD20'] = '{:.2f}'.format(pgd20_accuracy.mean)
        pbar_dic['PGD100'] = '{:.2f}'.format(pgd100_accuracy.mean)
        pbar_dic['MIM'] = '{:.2f}'.format(mim_accuracy.mean)
        pbar_dic['CW'] = '{:.2f}'.format(cw_accuracy.mean)
        pbar_dic['APGD_ce'] = '{:.2f}'.format(APGD_ce_accuracy.mean)
        pbar_dic['APGD_dlr'] = '{:.2f}'.format(APGD_dlr_accuracy.mean)
        pbar_dic['APGD_t'] = '{:.2f}'.format(APGD_t_accuracy.mean)
        pbar_dic['FAB_t'] = '{:.2f}'.format(FAB_t_accuracy.mean)
        pbar_dic['Square'] = '{:.2f}'.format(Square_accuracy.mean)
        pbar_dic['AA'] = '{:.2f}'.format(aa_accuracy.mean)
        pbar.set_postfix(pbar_dic)

    return [clean_accuracy.mean, pgd20_accuracy.mean, pgd100_accuracy.mean, mim_accuracy.mean, cw_accuracy.mean,
            APGD_ce_accuracy.mean, APGD_dlr_accuracy.mean, APGD_t_accuracy.mean, FAB_t_accuracy.mean,
            Square_accuracy.mean, aa_accuracy.mean]



