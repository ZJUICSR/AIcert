import numpy as np
import torch
from torch import optim
import time
import sys
# sys.path.append("..")
# sys.path.append("../..")
import os

from function.ensemble.CAFD.networks.denoiser import Denoiser

from function.ensemble.CAFD.processor import AverageMeter, accuracy

from function.ensemble.CAFD.dataload import DatasetIMG_Dual, DatasetNPY_Dual
from torchvision import transforms

import math
from function.ensemble.CAFD.example_cam import cam_divide_criteria, get_last_conv_name, getNetwork, CAM_divide_tensor
from function.ensemble.CAFD.networks.networks_NRP import Discriminator
from function.ensemble.attack.gen_adv import get_cafd_adv_dataloader

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


def save_checkpoint(state, save_dir, base_name="best_model"):
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


def train(epoch, denoiser, netD, target_model, learning_rate, weight_decay, dataloader, device, weight_adv, weight_act,
          weight_weight, ACT_stable, BCE_stable):
    denoiser.train()
    netD.train()

    optimizer = optim.Adam(denoiser.parameters(), lr=adjust_learning_rate(learning_rate, epoch),
                           weight_decay=weight_decay)

    optimizer_D = optim.Adam(netD.parameters(), lr=adjust_learning_rate(learning_rate, epoch),
                           weight_decay=weight_decay)


    losses = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    netD = netD.to(device)
    denoiser = denoiser.to(device)
    target_model = target_model.to(device)
    ACT_stable = ACT_stable.to(device)
    BCE_stable = BCE_stable.to(device)
    
    for i, (x, x_adv, y) in enumerate(dataloader):

        t_real = torch.ones((x.size(0), 1))
        t_fake = torch.zeros((x.size(0), 1))
        x, x_adv, y = x.to(device), x_adv.to(device), y.to(device)
        t_real, t_fake = t_real.to(device), t_fake.to(device)

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
                    BCE_stable(y_pred_fake - torch.mean(y_pred), t_real)) / 2) * weight_adv

        # Get logits from smooth and denoised image
        logits_smooth= target_model(x_smooth)

        # Compute loss

        loss_act, loss_weight = ACT_stable(x_smooth, x)

        loss_act = loss_act * weight_act

        loss_weight = loss_weight * weight_weight
        
        # loss_mse = MSE_stable(x_smooth, x) * args.weight_mse
        loss_adv, loss_act, loss_weight = loss_adv.to(device), loss_act.to(device), loss_weight.to(device)

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

        if i % 10 == 0:
            print('Train-Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(dataloader), batch_time=batch_time,
                loss=losses, top1=top1))


def test(denoiser, target_model, dataloader, device):

    denoiser.eval()

    batch_time = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    with torch.no_grad():
        for i, (x, x_adv, y) in enumerate(dataloader):
            x, x_adv, y = x.to(device), x_adv.to(device), y.to(device)

            # Compute denoised image. 
            noise = denoiser.forward(x_adv)
            x_smooth = x_adv + noise

            # Get logits from smooth and denoised image
            logits_smooth= target_model(x_smooth)


            prec1 = accuracy(logits_smooth.data, y)
            top1.update(prec1.item(), x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg

def cafd(target_model=None, dataloader=None, method='fgsm', adv_param={'eps':1}, channel=3, data_size=32, weight_adv=5e-3, weight_act=1e3, weight_weight=1e-3, lr=0.001, itr=70,
         batch_size=128, weight_decay=2e-4, print_freq=10, save_freq=2, device='cuda', gen_adv=True, dataset="mnist"):
    learning_rate = lr
    batch_size = batch_size
    num_epochs = itr
    print_freq = print_freq
    if type(device) == str:
        torch.device(device)
    use_cuda = True if device.type == 'cuda' else False
    # use_cuda = device == 'cuda'
    
    if gen_adv:
        train_loader, test_loader = get_cafd_adv_dataloader(method=method, model=target_model, dataloader=dataloader, attackparam=adv_param, device=device, batch_size=batch_size, dataset=dataset)
    else:
        train_loader, test_loader = dataloader, dataloader
    # Load Denoiser
    denoiser = Denoiser(x_h=data_size, x_w=data_size, channel=channel)
    # denoiser = NRP(3,3,64,5)

    # Load Discriminator
    netD = Discriminator(num_in_ch=channel, num_feat=32)

    if use_cuda:
        print(">>> SENDING MODEL TO GPU...")
        gpuid = int(str(device).split(':')[1])
        # denoiser.cuda()
        denoiser.to(device)
        denoiser = torch.nn.DataParallel(denoiser, device_ids=[gpuid]).to(device)

        # target_model.cuda()
        target_model.to(device)
        target_model = torch.nn.DataParallel(target_model, device_ids=[gpuid]).to(device)

        # netD.cuda()
        netD.to(device)
        netD = torch.nn.DataParallel(netD, device_ids=[gpuid]).to(device)

    target_model.eval()

    # load loss
    layer_name = get_last_conv_name(target_model)
    ACT_stable = cam_divide_criteria(CAM_divide_tensor(target_model, layer_name)).to(device)
    MSE_stable = torch.nn.MSELoss().to(device)
    BCE_stable = torch.nn.BCEWithLogitsLoss().to(device)

    # best_pred = 0.0
    # worst_pred = float("inf")
    # Prec = 0
    elapsed_time = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        train(epoch, denoiser, netD, target_model, learning_rate, weight_decay, train_loader,
              device, weight_adv, weight_act, weight_weight, ACT_stable, BCE_stable)
        prec = test(denoiser, target_model, test_loader, device)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' %(get_hms(elapsed_time)))
        if prec > 0.95:
            break
    print(f'prec={prec}')
    return {'denoiser': denoiser, 'prec': prec}


def denoising(denoiser, data):
    noise = denoiser(data)
    x_smooth = data + noise
    return x_smooth


if __name__ == '__main__':
    from function.ensemble.datasets.mnist import mnist_dataloader
    from function.ensemble.models.load_model import load_model
    methods = ['fgsm', 'bim', 'rfgsm', 'cw', 'pgd', 'tpgd', 'mifgsm', 'autopgd', 'square', 'deepfool', 'difgsm']
    # methods = ['mifgsm', 'fgsm', 'bim', 'rfgsm', 'cw', 'pgd', 'tpgd', 'autopgd', 'square', 'difgsm']
    # method = 'fgsm'
    methods = ['FGSM']
    device ='cuda'
    eps = 1
    model = load_model()
    model.to(device)
    _, dataloader= mnist_dataloader()
    results = dict()
    for method in methods:
        results[method] = cafd(target_model=model, dataloader=dataloader, method=method, eps=1, channel=1, data_size=28, weight_adv=5e-3,
         weight_act=1e3, weight_weight=1e-3, lr=0.001, itr=10,
         batch_size=128, weight_decay=2e-4, print_freq=10, save_freq=2, device=device)
    print(results)
    # 保存模型
    # torch.save(res['denoiser'], cafd.pt)
