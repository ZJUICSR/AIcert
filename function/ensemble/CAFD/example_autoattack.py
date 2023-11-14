import matplotlib.pyplot as plt
import os
from advertorch.utils import predict_from_logits
from torch.utils.data import DataLoader
from advertorch_examples.utils import bchw2bhwc
import torchvision.transforms as transforms
from autoattack import AutoAttack
import torchvision
import cv2
import pickle
import argparse
from networks import *
import torch.backends.cudnn as cudnn


num_classes = 10

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

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10')
parser.add_argument('--net_type', default='vggnet', type=str, help='model')
parser.add_argument('--depth', default=19, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=20, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
args = parser.parse_args()

cnt = 0

path_advu = './data/test/adv/AA/npy'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 500
transform_test = transforms.Compose([
    transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)

print('\n[Test Phase] : Model setup')
assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
_, file_name = getNetwork(args)
checkpoint = torch.load('./checkpoint/'+file_name+'.t7')
net = checkpoint['net']

if use_cuda:
    # net = BalancedDataParallel(50, net, dim=0).cuda()
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

net.eval()
net.training = False

label_true = []
label_cln = []
label_un = []

for cln_data, true_label in test_loader:

    cln_data, true_label = cln_data.to(device), true_label.to(device)

    adversary = AutoAttack(net, norm='Linf', eps=8/255, version='standard')

    for x in true_label:
        label_true.append(x.item())

    adv_untargeted = adversary.run_standard_evaluation(cln_data, true_label, bs=batch_size)

    pred_untargeted_adv = predict_from_logits(net(adv_untargeted))

    for x in pred_untargeted_adv:
        label_un.append(x.item())
    num1 = 0
    for n in range(batch_size):
        if pred_untargeted_adv[n] == true_label[n]:
            num1 += 1

    print("non-target:" + str(num1 / batch_size))

    for n in range(batch_size):
        cnt += 1

        img = bchw2bhwc(adv_untargeted[n].detach().cpu().numpy())
        # if img.shape[2] == 1:
        # img = np.repeat(img, 3, axis=2)

        name = str(cnt) + '.npy'
        # name = str(cnt) + '.png'
        path = os.path.join(path_advu, name)
        np.save(path, img)

        # img = cv2.resize(img, (28, 28))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(path, img * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    print(cnt)


