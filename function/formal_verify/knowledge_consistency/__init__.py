

import torch,os
def load_checkpoint(resume, model):
    if os.path.isfile(resume):
        # print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        state_dict = checkpoint['state_dict']
        keys = list(state_dict.keys())
        for key in keys:
            if key.find('module') >= 0:
                state_dict[key.replace('module.','')] = state_dict.pop(key)

        model.load_state_dict(state_dict)
        # print("=> loaded checkpoint '{}' (epoch {} acc1 {})"
        #       .format(resume, checkpoint['epoch'], checkpoint['best_acc1']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

def ResBlock_beforeReLU(block, x): # Only for ResNet 50 101 152!!
    identity = x

    out = block.conv1(x)
    out = block.bn1(out)
    out = block.relu(out)

    out = block.conv2(out)
    out = block.bn2(out)
    if block.__class__.__name__ != 'BasicBlock':
        out = block.relu(out)

        out = block.conv3(out)
        out = block.bn3(out)

    if block.downsample is not None:
        identity = block.downsample(x)

    out += identity
    return out

def get_feature(img, net, arch, conv_layer):
    input = img.unsqueeze(0)
    net.eval()
    with torch.no_grad():
        if arch.startswith("alexnet"):
            x = net.features[:conv_layer + 1](input)

        elif arch.startswith("vgg"):
            x = net.features[:conv_layer + 1](input)


        elif arch.startswith("resnet"):
            x = input
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            x = net.maxpool(x)

            x = net.layer1(x)
            x = net.layer2(x)
            x = net.layer3[:-1](x)
            x = ResBlock_beforeReLU(net.layer3[-1], x)

        return x.squeeze()