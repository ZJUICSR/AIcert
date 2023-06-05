from dc_util import *
from Datasets import Generate_Dataloader

parser = argparse.ArgumentParser(description='Get Variance')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--arch_in',   default='vgg16_bn')
parser.add_argument('--arch_tar',  default='vgg16_bn')
parser.add_argument('--optim', default='SGD',type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--dataset', default='VOC2012_crop', type=str)
parser.add_argument('--sample_num',default='', type = str)
parser.add_argument('--net_in', default='', type=str)
parser.add_argument('--net_tar', default='', type=str)
parser.add_argument('--transnet', default='', type=str)
parser.add_argument('--conv_layer', default=30, type=int)

args = parser.parse_args()
print('parsed options:', vars(args))

def get_variance(net, transNet, net2, dataloader):
    meter = AverageMeter()
    net.eval()
    transNet.eval()
    num_batches =  len(dataloader)
    with torch.no_grad():
        for i, (datas, target) in tqdm(enumerate(dataloader)):
            batch_size = target.size(0)
            x = conv_feature(datas, net, args.conv_layer,args.arch_in, args.gpu)
            out, out_n = transNet.val_batch(x)
            mean_batch = out_n.mean([1,3,4])
            meter.update(mean_batch, batch_size)
        
        Mean_all = meter.avg.reshape(3,1,-1,1,1)
        
        meter.reset()
        for i, (datas, target) in tqdm(enumerate(dataloader)):
            batch_size = target.size(0)
            x = conv_feature(datas,net, args.conv_layer,args.arch_in, args.gpu)
            out, out_n = transNet.val_batch(x)
            var_batch = ((out_n-Mean_all)**2).mean([1,2,3,4])
            meter.update(var_batch, batch_size)  
        
        Mean_all = Mean_all.reshape(3,-1)
        Var_all = meter.avg
        
        print(Var_all)
        meter.reset()
        for i, (datas, target) in tqdm(enumerate(dataloader)):
            batch_size = target.size(0)
            t = conv_feature(datas, net2, args.conv_layer,args.arch_tar, args.gpu)
            x = conv_feature(datas, net, args.conv_layer,args.arch_in, args.gpu)
            out, out_n = transNet.val_batch(x)
            diff = t - out
            d_mean = diff.mean([0,2,3])
            meter.update(d_mean, batch_size)
        diff_mean = meter.avg.reshape(1, -1, 1, 1)
        
        meter.reset()
        for i, (datas, target) in tqdm(enumerate(dataloader)):
            t = conv_feature(datas, net2, args.conv_layer,args.arch_tar, args.gpu)
            x = conv_feature(datas, net, args.conv_layer,args.arch_in, args.gpu)
            out, out_n = transNet.val_batch(x)
            diff = t - out
            diff_var = ((diff-diff_mean)**2).mean([0,1,2,3])
            meter.update(diff_var, batch_size) 
        diff_var = meter.avg
        print(diff_var) 
        
        return Mean_all, Var_all, diff_var


def main():
    if args.arch_in.startswith("vgg16_bn"):
        channels = 512
        kernel_size = 28 if args.conv_layer<=30 else 14
    elif args.arch_in.startswith("alexnet"):
        channels = 256
        kernel_size = 13
    elif args.arch_in.startswith('resnet'):
        if args.arch_in.startswith('resnet18') or args.arch_in.startswith('resnet34'):
            if args.conv_layer == 3:
                channels = 256
                kernel_size = 14
            elif args.conv_layer == 4:
                channels = 512
                kernel_size = 7
        else:
            if args.conv_layer == 3:
                channels = 1024
                kernel_size = 14
            elif args.conv_layer == 4:
                channels = 2048
                kernel_size = 7
    # Load models
    if args.dataset.startswith("VOC"):
        net_in = models.__dict__[args.arch_in](num_classes=20)
        net_tar = models.__dict__[args.arch_tar](num_classes=20)
    elif args.dataset.startswith("CUB"):
        net_in = models.__dict__[args.arch_in](num_classes=200)
        net_tar = models.__dict__[args.arch_tar](num_classes=200)
    elif args.dataset.startswith("DOG"):
        net_in = models.__dict__[args.arch_in](num_classes=200)
        net_tar = models.__dict__[args.arch_tar](num_classes=200)

    # load vgg model
    print("load model_vgg......")

    if args.arch_in.endswith('DC'):
        net_in = load_masked_model_as_original_model(args.net_in)
    else:
        load_checkpoint(args.net_in, net_in)

    if args.arch_tar.endswith('DC'):
        net_tar = load_masked_model_as_original_model(args.net_tar)
    else:
        load_checkpoint(args.net_tar, net_tar)
    


    # load 3 layers model
    print("load model_Ys......")
    #######################################################
    if args.arch_in.startswith("vgg"):
        if args.conv_layer <= 30:
            input_size = output_size = torch.zeros((512, 28, 28)).shape
        else:
            input_size = output_size = torch.zeros((512, 14, 14)).shape
    elif args.arch_in.startswith("alexnet"):
        input_size = output_size = torch.zeros((256, 13, 13)).shape
    elif args.arch_in.startswith("resnet"):
        if args.arch_in.startswith('resnet18') or args.arch_in.startswith('resnet34'):
            input_size = output_size = torch.zeros((256, 14, 14)).shape   
        else:
            input_size = output_size = torch.zeros((1024, 14, 14)).shape
    #######################################################
    model_Ys = models.LinearTester(input_size, output_size, gpu_id=args.gpu, fix_p=True, bn=False, instance_bn=True)
    load_checkpoint(args.transnet, model_Ys)
    model_Ys = model_Ys.cpu()

    net_in.cuda(args.gpu)
    net_tar.cuda(args.gpu)
    model_Ys.cuda(args.gpu)

    # Create dataloader
    train_loader, val_loader = \
        Generate_Dataloader(args.dataset, args.batch_size, args.workers,
                        '', args.sample_num)

    print(get_variance(net_in, model_Ys, net_tar, val_loader))

if __name__ == '__main__':
    main()