import matplotlib.pyplot as plt
import os
from advertorch.utils import predict_from_logits
from torch.utils.data import DataLoader
from advertorch_examples.utils import bchw2bhwc
import torchvision

import torchvision.transforms as transforms
from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, DDNL2Attack, SpatialTransformAttack

import argparse
from networks import *
import torch.backends.cudnn as cudnn
from utils.BalancedDataParallel import BalancedDataParallel


def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

plt.switch_backend('agg')

torch.manual_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='vggnet', type=str, help='model')
parser.add_argument('--depth', default=19, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=20, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
args = parser.parse_args()

cnt = 0
path_cln = './data/test/clean/npy'
path_advu = './data/test/adv/PGD/npy'
path_advt = './data/test/adv/PGD_t/npy'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 500
transform_test = transforms.Compose([
    transforms.ToTensor()])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

num_classes = 10
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)

print('\n[Test Phase] : Model setup')
assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
_, file_name = getNetwork(args)
checkpoint = torch.load('./checkpoint/'+file_name+'.t7')
net = checkpoint['net']

if use_cuda:
    # net = BalancedDataParallel(1, net, dim=0).cuda()
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

net.eval()
net.training = False

label_true = []

for cln_data, true_label in test_loader:

    cln_data, true_label = cln_data.to(device), true_label.to(device)
    for x in true_label:
        label_true.append(x.item())


    '''
    adversary = SpatialTransformAttack(
        net, 10, clip_min=0.0, clip_max=1.0, max_iterations=100, search_steps=20, targeted=False)
    '''


    adversary = LinfPGDAttack(
    net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)

    '''
    adversary = CarliniWagnerL2Attack(
        net, 10, clip_min=0.0, clip_max=1.0, max_iterations=100, confidence=1, initial_const=1, learning_rate=1e-2,
        binary_search_steps=4, targeted=False)
    '''

    '''
    adversary = DDNL2Attack(net, nb_iter=100, gamma=0.05, init_norm=1.0, quantize=True, levels=256, clip_min=0.0,
                            clip_max=1.0, targeted=False, loss_fn=None)
    '''

    adv_untargeted = adversary.perturb(cln_data, true_label)
    target = torch.ones_like(true_label) * 3
    for x in range(len(true_label)):
        if target[x] == true_label[x]:
            target[x] = 1

    adversary.targeted = True
    adv_targeted = adversary.perturb(cln_data, target)

    pred_cln = predict_from_logits(net(cln_data))
    pred_untargeted_adv = predict_from_logits(net(adv_untargeted))
    pred_targeted_adv = predict_from_logits(net(adv_targeted))

    num1 = 0
    num2 = 0
    for n in range(len(true_label)):
        if pred_untargeted_adv[n] == true_label[n]:
            num1 += 1
        if pred_targeted_adv[n] == true_label[n]:
            num2 += 1
    print("non-target:" + str(num1/batch_size))
    print("target:" + str(num2 / batch_size))

    for n in range(len(true_label)):
        cnt += 1

        img = bchw2bhwc(cln_data[n].detach().cpu().numpy())
        #if img.shape[2] == 1:
            #img = np.repeat(img, 3, axis=2)

        #name = str(cnt) + '.png'
        name = str(cnt) + '.npy'
        path = os.path.join(path_cln, name)
        np.save(path, img)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(path, img*255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # adv_untargeted-------------------------------------------------
        img = bchw2bhwc(adv_untargeted[n].detach().cpu().numpy())

        #name = str(cnt) + '.png'
        name = str(cnt) + '.npy'
        path = os.path.join(path_advu, name)
        # img = cv2.resize(img, (28, 28))
        np.save(path, img)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(path, img*255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


        # adv_targeted-------------------------------------------------
        img = bchw2bhwc(adv_targeted[n].detach().cpu().numpy())

        # name = str(cnt) + '.png'
        name = str(cnt) + '.npy'
        path = os.path.join(path_advt, name)
        # img = cv2.resize(img, (28, 28))
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(path, img*255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        np.save(path, img)
    print(cnt)

pickle.dump(label_true, open('./data/test/clean/label_true.pkl', 'wb'))


