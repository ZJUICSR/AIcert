import numpy as np
import torch
from torch import optim

import argparse
import time

import os

from networks.denoiser import Denoiser

from processor import AverageMeter, accuracy

import math
from os.path import join
from example_cam import cam_divide_criteria, get_last_conv_name, getNetwork, CAM_divide_tensor
from networks.networks_NRP import DiscriminatorCifar
from cifar_data import get_cifar_fuse_dataloader
from networks import Wide_ResNet

irange = range


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def adjust_learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 60):
        optim_factor = 3
    elif(epoch > 50):
        optim_factor = 2
    elif(epoch > 40):
        optim_factor = 1

    return init*math.pow(0.3, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """


    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    return grid

def save_checkpoint(state, save_dir, base_name="cifar_best_model"):
    """Saves checkpoint to disk"""
    directory = save_dir
    filename = base_name + ".pth.tar"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    #im = im.convert('L')
    im.save(filename, quality=100)
    
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training. See code for default values.')

# STORAGE LOCATION VARIABLES
parser.add_argument('--save_dir', '--sd', default='./checkpoint_denoise/CAFD/', type=str, help='Path to Model')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--Tcheckpoint', default='./checkpoint')
parser.add_argument('--layer-name', type=str, default=None, help='last convolutional layer name')
# parser.add_argument('--weight_mse', default=0, type=float, help='weight_mse 0.1')
parser.add_argument('--weight_adv', default=5e-3, type=float, help='weight_adv 0.001')
parser.add_argument('--weight_act', default=1e3, type=float, help='weight_act')
parser.add_argument('--weight_weight', default=1e-3, type=float, help='weight_lable')


# MODEL HYPERPARAMETERS
parser.add_argument('--lr', default=0.001, metavar='lr', type=float, help='Learning rate')
parser.add_argument('--itr', default=30, metavar='iter', type=int, help='Number of iterations')
parser.add_argument('--batch_size', default=200, metavar='batch_size', type=int, help='Batch size')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, help='weight decay (default: 2e-4)')
parser.add_argument('--print_freq', '-p', default=75, type=int, help='print frequency (default: 10)')
parser.add_argument('--save_freq', '-sf', default=2, type=int, help='print frequency (default: 10)')


# OTHER PROPERTIES
parser.add_argument('--gpu', default="0,1", type=str, help='GPU devices to use (0-7) (default: 0,1)')
parser.add_argument('--mode', default=0, type=int, help='Wether to perform test without trainig (default: 0)')
parser.add_argument('--path_denoiser', default='./checkpoint_denoise/CAFD/cifar/best_model.pth.tar', type=str, help='Denoiser path')
parser.add_argument('--saveroot', default='./results/defense/adv/PGD', type=str, help='output images')

parser.add_argument('--attack_method', default='fgsm', type=str, help='attack method')
parser.add_argument('--eps', default=2, type=int, help='eps')

args = parser.parse_args()

setup_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# Other Variables

TRAIN_AND_TEST = 0
TEST = 1

save_dir = args.save_dir
start_epoch = 1

# Set Model Hyperparameters
learning_rate = args.lr
batch_size = args.batch_size
num_epochs = args.itr
print_freq = args.print_freq
use_cuda = torch.cuda.is_available()


train_loader, test_loader = get_cifar_fuse_dataloader(method=args.attack_method,
                                                      eps=args.eps,
                                                      batch_size=batch_size,
                                                      train_total=20000,
                                                      test_total=5000)

# Load Denoiser
denoiser = Denoiser(x_h=32, x_w=32)
# denoiser = NRP(3,3,64,5)

# Load Discriminator
netD = DiscriminatorCifar(3, 32)

# Load Target Model
print('\n[Test Phase] : Model setup')
assert os.path.isdir(args.Tcheckpoint), 'Error: No Tcheckpoint directory found!'


target_model = Wide_ResNet(depth=16, widen_factor=10, dropout_rate=0.3, num_classes=10)
checkpoints = torch.load("/opt/data/user/gss/code/cyber_adv/Comdefend/checkpoints/cifar10/wide-resnet-16x10.pt")

target_model.load_state_dict(checkpoints['model'])


if use_cuda:
    print(">>> SENDING MODEL TO GPU...")
    denoiser.cuda()
    denoiser = torch.nn.DataParallel(denoiser, device_ids=range(torch.cuda.device_count()))

    target_model.cuda()
    target_model = torch.nn.DataParallel(target_model, device_ids=range(torch.cuda.device_count()))

    netD.cuda()
    netD = torch.nn.DataParallel(netD, device_ids=range(torch.cuda.device_count()))

target_model.eval()

# load loss
layer_name = get_last_conv_name(target_model) if args.layer_name is None else args.layer_name
ACT_stable = cam_divide_criteria(CAM_divide_tensor(target_model, layer_name)).cuda()
MSE_stable = torch.nn.MSELoss().cuda()
BCE_stable = torch.nn.BCEWithLogitsLoss().cuda()

best_pred = 0.0
worst_pred = float("inf")


def train(epoch):
    denoiser.train()
    netD.train()

    optimizer = optim.Adam(denoiser.parameters(), lr=adjust_learning_rate(learning_rate, epoch),
                           weight_decay=args.weight_decay)

    optimizer_D = optim.Adam(netD.parameters(), lr=adjust_learning_rate(learning_rate, epoch),
                           weight_decay=args.weight_decay)


    losses = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    for i, (x, x_adv, y) in enumerate(train_loader):

        t_real = torch.ones((x.size(0), 1))
        t_fake = torch.zeros((x.size(0), 1))
        if use_cuda:
            x, x_adv, y = x.cuda(), x_adv.cuda(), y.cuda()
            t_real, t_fake = t_real.cuda(), t_fake.cuda()

        # train netD
        y_pred = netD(x)
        noise = denoiser.forward(x_adv).detach()
        x_smooth = x_adv + noise
        y_pred_fake = netD(x_smooth)

        loss_D = (BCE_stable(y_pred - torch.mean(y_pred_fake), t_real) +
                  BCE_stable(y_pred_fake - torch.mean(y_pred), t_fake)) / 2

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Compute denoised image. 
        noise = denoiser.forward(x_adv)
        x_smooth = x_adv + noise

        # adv_loss
        y_pred = netD(x)
        y_pred_fake = netD(x_smooth)

        loss_adv = ((BCE_stable(y_pred - torch.mean(y_pred_fake), t_fake) +
                    BCE_stable(y_pred_fake - torch.mean(y_pred), t_real)) / 2) * args.weight_adv

        # Get logits from smooth and denoised image
        logits_smooth= target_model(x_smooth)
 

        # Compute loss

        loss_act, loss_weight = ACT_stable(x_smooth, x)

        loss_act = loss_act * args.weight_act

        loss_weight = loss_weight * args.weight_weight
        
        # loss_mse = MSE_stable(x_smooth, x) * args.weight_mse

        loss = loss_adv + loss_act + loss_weight #   loss_mse

        # Update Mean loss for current iteration

        losses.update(loss.item(), x.size(0))
        prec1 = accuracy(logits_smooth.data, y)
        top1.update(prec1.item(), x.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # Set grads to zero for new iter
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Train-Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))


def test(epoch, args):

    denoiser.eval()
    netD.eval()

    batch_time = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    with torch.no_grad():
        for i, (x, x_adv, y) in enumerate(test_loader):

            if use_cuda:
                x, x_adv, y = x.cuda(), x_adv.cuda(), y.cuda()

            # Compute denoised image. 
            noise = denoiser.forward(x_adv)
            x_smooth = x_adv + noise


            # Get logits from smooth and denoised image
            logits_smooth= target_model(x_smooth)


            prec1 = accuracy(logits_smooth.data, y)
            top1.update(prec1.item(), x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test-Epoch: [{0}][{1}/{2}]'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(test_loader), batch_time=batch_time, top1=top1))

                out = torch.stack((x, x_smooth))  # 2, bs, 3, 32, 32
                out = out.transpose(1, 0).contiguous()  # bs, 2, 3, 32, 32
                out = out.view(-1, x.size(-3), x.size(-2), x.size(-1))
                if not os.path.exists('./checkpoint_denoise'):
                    os.makedirs('./checkpoint_denoise')
                # save_image(out, join('./checkpoint_denoise', 'test_recon_{}.png'.format(i)), nrow=20)

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

        if epoch % args.save_freq == 0:
            save_checkpoint(denoiser.state_dict(), save_dir, base_name=f'cifar10_{args.attack_method}_{args.eps}_cafd')
            print('save the model')


def evaluate(path_denoiser, saveroot):
    cnt = 0
    denoiser.load_state_dict(torch.load(path_denoiser))
    denoiser.eval()

    top1 = AverageMeter()

    if not os.path.exists(saveroot):
        os.makedirs(saveroot)

    for i, (_, x_adv, y) in enumerate(test_loader):

        if use_cuda:
            x_adv, y = x_adv.cuda(), y.cuda()

        noise = denoiser.forward(x_adv)
        x_smooth = x_adv + noise

        logits_smooth = target_model(x_smooth)
        prec1 = accuracy(logits_smooth.data, y)
        top1.update(prec1.item(), x_adv.size(0))

        for n in range(x_smooth.size(0)):
            cnt += 1
            out = torch.unsqueeze(x_smooth[n], 0)

            save_image(out, join(saveroot, '{}.png'.format(cnt)), nrow=1, padding=0)

    print(' * Prec@1 {top1.avg:.4f}'.format(top1=top1))


if args.mode == TRAIN_AND_TEST:
    print("==================== TRAINING ====================")
    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(learning_rate))

    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch+num_epochs):
        start_time = time.time()

        train(epoch)
        test(epoch, args)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' %(get_hms(elapsed_time)))

    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.4f' %(best_pred))

if args.mode == TEST:
    print("==================== TESTING ====================")
    evaluate(args.path_denoiser, args.saveroot)
