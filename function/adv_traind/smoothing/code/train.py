import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
import time
import datetime
from train_utils import AverageMeter, accuracy, init_logfile, log


def main(args_dict):
    if args_dict['gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = args_dict['gpu']

    if not os.path.exists(args_dict['outdir']):
        os.mkdir(args_dict['outdir'])

    train_dataset = get_dataset(args_dict['dataset'], 'train')
    test_dataset = get_dataset(args_dict['dataset'], 'test')
    pin_memory = (args_dict['dataset'] == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args_dict['batch'],
                              num_workers=args_dict['workers'], pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args_dict['batch'],
                             num_workers=args_dict['workers'], pin_memory=pin_memory)

    model = get_architecture(args_dict['arch'], args_dict['dataset'])

    logfilename = os.path.join(args_dict['outdir'], 'log.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")

    criterion = CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), lr=args_dict['lr'], momentum=args_dict['momentum'],
                    weight_decay=args_dict['weight_decay'])
    scheduler = StepLR(optimizer, step_size=args_dict['lr_step_size'], gamma=args_dict['gamma'])

    for epoch in range(args_dict['epochs']):
        scheduler.step(epoch)
        before = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args_dict['noise_sd'])
        test_loss, test_acc = test(test_loader, model, criterion, args_dict['noise_sd'])
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))

        torch.save({
            'epoch': epoch + 1,
            'arch': args_dict['arch'],
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args_dict['outdir'], 'checkpoint.pth.tar'))
        return train_acc, test_acc


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()

        # augment inputs with noise
        inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args_dict['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg


def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args_dict['print_freq'] == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        return acc1, acc5


if __name__ == "__main__":
    args_dict = {
        'dataset': 'cifar10',
        'arch': 'cifar_resnet110',
        'outdir': "/data/user/WZT/models/smoothing/cifar10/resnet110/noise_0.25",
        'workers': 4,
        'epochs': 90,
        'batch': 256,
        'lr': 0.1,
        'lr_step_size': 30,
        'gamma': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'noise_sd': 0.25,
        'gpu': 0,
        'print_freq': 10
    }
    main(args_dict)
