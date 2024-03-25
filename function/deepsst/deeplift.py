import torch
import torch.nn.functional as F
import os
import time
import gc


def mappingtable(kernel_size, stride, padding, x, y):
    xshape = x.shape
    if isinstance(padding, tuple):
        paddingu = padding[0]
        paddingv = padding[1]
    else:
        paddingu = paddingv = padding
    x = torch.nn.ZeroPad2d((paddingu, paddingu, paddingv, paddingv))(x)
    if isinstance(stride, tuple):
        strideu = stride[0]
        stridev = stride[1]
    else:
        strideu = stridev = stride
    if isinstance(kernel_size, tuple):
        kernel_sizeu = kernel_size[0]
        kernel_sizev = kernel_size[1]
    else:
        kernel_sizeu = kernel_sizev = kernel_size
    dict = {}

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            dict[(i, j)] = [[u - paddingu, v - paddingv] for u in range(i * strideu, i * strideu + kernel_sizeu)
                            if - 1 < u - paddingu < xshape[0]
                            for v in range(j * stridev, j * stridev + kernel_sizev)
                            if - 1 < v - paddingv < xshape[1]]

    return dict


def get_M(net, activation, name, ref_x, ref_y, debug=False):
    def hook_FC(model, input, output):  # 层间贡献率 FC
        if debug:
            modelname = name
            print(modelname)
            tm = time.time()
        deltax = input[0] - ref_x  # n,in
        deltax = deltax.to(device)
        signs = torch.zeros((deltax.shape[0], deltax.shape[1], model.weight.shape[0])).to(device)
        # print(signs.shape)
        for n in range(deltax.shape[0]):
            signs[n] = torch.sign(deltax[n] * model.weight).T
        del deltax
        signs[signs == 0] = 0.5
        signs_p = signs.clone()
        signs_p[signs_p == -1] = 0
        signs_p = torch.sum(signs_p, dim=0).to(device)

        signs_n = signs
        signs_n[signs_n == 1] = 0
        signs_n[signs_n == -1] = 1

        signs_n = torch.sum(signs_n, dim=0).to(device)

        activation[name + '_mpp'] = (model.weight * signs_p.T).T
        activation[name + '_mnn'] = (model.weight * signs_n.T).T
        del signs_p
        del signs_n
        del signs
        torch.cuda.empty_cache()
        gc.collect()
        if debug:
            print('use time: '+str(time.time()-tm))

    def hook_activations(model, input, output):
        # modelname = name
        # print(modelname)
        # tm = time.time()
        deltax = input[0] - ref_x  # n,l
        deltay = output - ref_y  # n,l
        m = deltay / deltax  # n,l
        m = m.to(device)
        m = torch.sum(m, dim=0)  # 待改
        if len(m.shape) == 1:
            activation[name + '_m'] = torch.eye(m.shape[0]).to(device) * m
        elif len(m.shape) == 3:
            activation[name + '_m'] = torch.zeros(m.shape[0], m.shape[1], m.shape[2], m.shape[0], m.shape[1],
                                                  m.shape[2]).to(device)
            for a in range(m.shape[0]):
                for b in range(m.shape[1]):
                    for c in range(m.shape[2]):
                        activation[name + '_m'][a, b, c, a, b, c] = m[a, b, c]
            dm = m.shape[0] * m.shape[1] * m.shape[2]
            activation[name + '_m'] = activation[name + '_m'].reshape((dm, dm)).to(device)
        del m

        # print('use time: '+str(time.time()-tm))

    def hook_activations_new(model, input, output):
        if debug:
            modelname = name
            print(modelname)
            tm = time.time()
        deltax = input[0] - ref_x  # n,l
        deltay = output - ref_y  # n,l
        m = deltay / (deltax+1e-6)  # n,l
        m = torch.sum(m, dim=0)  # 待改
        if len(m.shape) == 1:
            activation[name + '_m'] = m.reshape(m.shape[0],1)
        elif len(m.shape) == 3:
            dm = m.shape[0] * m.shape[1] * m.shape[2]
            m = m.reshape(dm).to(device)
            activation[name + '_m'] = m.reshape(m.shape[0],1)
        #del m
        del deltax
        del deltay
        torch.cuda.empty_cache()
        gc.collect()
        if debug:
            print('use time: '+str(time.time()-tm))

    def hook_Conv(model, input, output):
        if debug:
            
            modelname = name
            print(modelname)
        # input: nbatch,din,x,y
        # weight: outputz,inputz,kernelx,kernely
        # output: nbatch,dout,x,y
            tm = time.time()
        deltax = input[0] - ref_x  # n,d,x,y
        # map_dict = mappingtable(model.kernel_size, model.stride, model.padding, input[0][0], output[0][0])
        mpp = torch.zeros((deltax.shape[0],
                           deltax.shape[1], deltax.shape[2] + 2 * model.padding[0],
                           deltax.shape[3] + 2 * model.padding[1],
                           output.shape[1], output.shape[2], output.shape[3]))
        mnn = torch.zeros((deltax.shape[0],
                           deltax.shape[1], deltax.shape[2] + 2 * model.padding[0],
                           deltax.shape[3] + 2 * model.padding[1],
                           output.shape[1], output.shape[2], output.shape[3]))
        deltax = torch.nn.ZeroPad2d((model.padding[0], model.padding[0], model.padding[1], model.padding[1]))(deltax)
        for n in range(deltax.shape[0]):
            for d in range(model.weight.shape[0]):
                for outx in range(output.shape[2]):
                    for outy in range(output.shape[3]):
                        xin_min = outx * model.stride[0]
                        xin_max = outx * model.stride[0] + model.kernel_size[0]
                        yin_min = outy * model.stride[1]
                        yin_max = outy * model.stride[1] + model.kernel_size[1]
                        # print('xmin' + str(xin_min) + 'xmax' + str(xin_max) + 'ymin' + str(yin_min) + 'ymax' + str(
                        #    yin_max))
                        # print(deltax[n, :, xin_min:xin_max, yin_min:yin_max].shape)
                        # print(model.weight[d].shape)
                        signs = torch.sign(deltax[n, :, xin_min:xin_max, yin_min:yin_max] * model.weight[d])
                        signs[signs == 0] = 0.5
                        signs_p = signs.clone()
                        signs_p[signs_p == -1] = 0

                        signs_n = signs
                        signs_n[signs_n == 1] = 0
                        signs_n[signs_n == -1] = 1
                        mpp[n, :, xin_min:xin_max, yin_min:yin_max, d, outx, outy] = model.weight[d] * signs_p
                        mnn[n, :, xin_min:xin_max, yin_min:yin_max, d, outx, outy] = model.weight[d] * signs_n
        mpp = torch.sum(mpp, dim=0)
        mpp = mpp[:, model.padding[0]:model.padding[0] + input[0].shape[2],
              model.padding[1]:model.padding[1] + input[0].shape[3], :, :, :]
        mpp = mpp.reshape((mpp.shape[0] * mpp.shape[1] * mpp.shape[2], mpp.shape[3] * mpp.shape[4] * mpp.shape[5]))
        mnn = torch.sum(mnn, dim=0)
        mnn = mnn[:, model.padding[0]:model.padding[0] + input[0].shape[2],
              model.padding[1]:model.padding[1] + input[0].shape[3], :, :, :]
        mnn = mnn.reshape((mnn.shape[0] * mnn.shape[1] * mnn.shape[2], mnn.shape[3] * mnn.shape[4] * mnn.shape[5]))
        activation[name + '_mpp'] = mpp
        activation[name + '_mnn'] = mnn
        if debug:
            print('use time: '+str(time.time()-tm))

    def hook_Conv_New(model, input, output):
        if debug:
            modelname = name
            print(modelname)
            tm = time.time()
        wp = model.weight * (model.weight > 0)
        wn = model.weight * (model.weight < 0)
        deltax = input[0] - ref_x  # n,in
        deltax = deltax.to(device)
        pconv = torch.nn.ConvTranspose2d(model.out_channels, model.in_channels, model.kernel_size,
                                         padding=model.padding, bias=False)
        nconv = torch.nn.ConvTranspose2d(model.out_channels, model.in_channels, model.kernel_size,
                                         padding=model.padding, bias=False)
        pconv.weight.data = wp
        nconv.weight.data = wn
        tconv = torch.nn.ConvTranspose2d(model.out_channels, model.in_channels, model.kernel_size,
                                         padding=model.padding, bias=False)
        tconv.weight = model.weight
        if debug:
            print('use time: '+str(time.time()-tm))
        activation[name + '_dx'] = deltax
        activation[name + '_tconv'] = tconv
        activation[name + '_pconv'] = pconv
        activation[name + '_nconv'] = nconv
        activation[name + '_yshape'] = output.shape
        del wp
        del wn
        torch.cuda.empty_cache()
        gc.collect()

    def hook_BN(model, input, output):
        if debug:
            modelname = name
            print(modelname)
            tm = time.time()
        activation[name + '_m'] = (model.weight / torch.sqrt(model.running_var)).repeat(
            input[0].shape[-1]*input[0].shape[-2]).unsqueeze(1)
        if debug:
            print('use time: '+str(time.time()-tm))

    def hook_maxpool(model, input, output):
        if debug:
            modelname = name
            print(modelname)
            tm = time.time()
        # input: nbatch,din,x,y
        # weight: outputz,inputz,kernelx,kernely
        # output: nbatch,dout,x,y
        # din = dout
        deltax = input[0] - ref_x  # n,d,x,y
        deltay = output - ref_y
        m = torch.zeros((deltax.shape[0],
                         deltax.shape[1], deltax.shape[2], deltax.shape[3],
                         output.shape[1], output.shape[2], output.shape[3]))
        map_dict = mappingtable(model.kernel_size, model.stride, model.padding, ref_x[0][0], ref_y[0][0])
        for n in range(deltax.shape[0]):
            for d in range(deltax.shape[1]):
                for (xout, yout), in_list in map_dict.items():
                    for (xin, yin) in in_list:
                        if deltax[n, d, xin, yin] != 0:
                            m[n, d, xin, yin, d, xout, yout] = deltay[n, d, xout, yout] / (
                                    deltax[n, d, xin, yin] * len(in_list))

                        else:
                            m[n, d, xin, yin, d, xout, yout] = 0
        m = torch.sum(m, dim=0)
        # print(np.sum(m[:,:,:,:,3:,:].detach().numpy()))
        m = m.reshape((m.shape[0] * m.shape[1] * m.shape[2], m.shape[3] * m.shape[4] * m.shape[5]))
        activation[name + '_m'] = m.to(device)
        del map_dict
        del deltax
        del deltay
        del m
        torch.cuda.empty_cache()
        gc.collect()
        if debug:
            print('use time: '+str(time.time()-tm))
    
    
    def hook_maxpool_new(model, input, output):
        if debug:
            modelname = name
            print(modelname)
            tm = time.time()
        deltax = input[0] - ref_x  # n,d,x,y
        deltay = output - ref_y
        if model.padding:
            deltax = torch.nn.ZeroPad2d(model.padding)(deltax)
        if isinstance(model.stride,tuple):
            up = torch.nn.Upsample(scale_factor=model.stride, mode='nearest')
            m = (up(deltay)/(deltax * model.stride[0] * model.stride[1] + 1e-6))[0]
        else:
            up = torch.nn.Upsample(scale_factor=(model.stride,model.stride), mode='nearest')
            m = (up(deltay)/(deltax * model.stride * model.stride + 1e-6))[0]
        if model.padding:
            m = m[:,model.padding:-model.padding,model.padding:-model.padding]
        m = m.reshape(m.shape[0]*m.shape[1]*m.shape[2],1)
        activation[name + '_m'] = m.to(device)
        def deal(m_next):
            m_next = up(m_next.reshape(output.shape[1],output.shape[2],output.shape[3],10).permute(-1,0,1,2))
            m_next = m_next.permute(1,2,3,0).reshape((input[0].shape[1]*input[0].shape[2]*input[0].shape[3],10))
            return m_next
        activation[name + '_up'] = deal
        del deltax
        del deltay
        if debug:
            print('use time: '+str(time.time()-tm))
        
        
    def hook_flatten(model, input, output):
        # modelname = name
        # print(modelname)
        # tm = time.time()
        m = torch.zeros((input[0].shape[0],
                         input[0].shape[1], input[0].shape[2], input[0].shape[3],
                         output.shape[1]))
        for s0 in range(input[0].shape[0]):
            for s1 in range(input[0].shape[1]):
                for s2 in range(input[0].shape[2]):
                    for s3 in range(input[0].shape[3]):
                        current = s1 * input[0].shape[1] + s2 * input[0].shape[2] + s3 * input[0].shape[3]
                        m[s0, s1, s2, s3, current] = 1
        m = torch.sum(m, dim=0)
        m = m.reshape((m.shape[0] * m.shape[1] * m.shape[2], m.shape[3]))
        activation[name + '_m'] = m
        # print('use time: '+str(time.time()-tm))

    def hook_flatten_new(model, input, output):
        if debug:
            modelname = name
            print(modelname)
            tm = time.time()
        activation[name + '_m'] = torch.eye(output.shape[1]).to(device)
        if debug:
            print('use time: '+str(time.time()-tm))

    if 'Linear' in net._modules[name]._get_name():
        return hook_FC
    elif 'ReLU' in net._modules[name]._get_name() or \
            'Sigmold' in net._modules[name]._get_name() or \
            'Tanh' in net._modules[name]._get_name():
        return hook_activations_new
    elif 'Conv' in net._modules[name]._get_name():
        return hook_Conv_New
    elif 'MaxPool' in net._modules[name]._get_name():
        if net._modules[name].stride == net._modules[name].kernel_size:
            return hook_maxpool_new
        else:
            return hook_maxpool
    elif 'Flatten' in net._modules[name]._get_name():
        return hook_flatten_new
    elif 'BatchNorm' in net._modules[name]._get_name():
        return hook_BN


def get_activation(activation, name):
    def hook(model, input, output):
        activation[name + '_input'] = input[0]
        activation[name + '_output'] = output

    return hook


def get_ref_output_for_layer(net, layer_name, ref_input):
    target_layer = net._modules[layer_name]
    ref_outputs = {}
    h = target_layer.register_forward_hook(get_activation(ref_outputs, layer_name))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    ref_input = ref_input.to(device)

    with torch.no_grad():
        output = net(ref_input).data
        target_input = ref_outputs[layer_name + '_input'].data
        target_output = ref_outputs[layer_name + '_output'].data
        del output
        torch.cuda.empty_cache()
    h.remove()

    return target_input.to(device), target_output.to(device)


def hook_reg(net, activation, ref_input):
    ref_x = {}
    ref_y = {}
    for layer in net._modules:
        if net._modules[layer]._get_name() != 'Sequential':
            ref_x[layer], ref_y[layer] = get_ref_output_for_layer(net, layer, ref_input)
        else:
            for sublayer in net._modules[layer]._modules:
                ref_x[sublayer], ref_y[sublayer] = get_ref_output_for_layer(net._modules[layer], sublayer, ref_input)
    for layer in net._modules:
        if net._modules[layer]._get_name() != 'Sequential':
            net._modules[layer].register_forward_hook(
                get_M(net, activation, layer, ref_x[layer].to(device), ref_y[layer].to(device),debug=False))
        else:
            for sublayer in net._modules[layer]._modules:
                net._modules[layer]._modules[sublayer].register_forward_hook(
                    get_M(net._modules[layer], activation, sublayer, ref_x[sublayer].to(device),
                          ref_y[sublayer].to(device),debug=False))


def deal_layer_m(net,activation,last_layer,true_last_layer,current_layer):
    if activation.get(current_layer + '_m') is not None:
        if 'ReLU' in net._modules[current_layer]._get_name() or 'BatchNorm' in net._modules[current_layer]._get_name():
            activation[current_layer + '_mpp'] = torch.mul(activation[current_layer + '_m'],
                                                           activation[last_layer + '_mpp'])
            activation[current_layer + '_mnn'] = torch.mul(activation[current_layer + '_m'],
                                                           activation[last_layer + '_mnn'])
        elif activation.get(current_layer + '_up'):
            activation[current_layer + '_mpp'] = torch.mul(activation[current_layer + '_m'],
                                    activation[current_layer + '_up'](activation[last_layer + '_mpp']))
            activation[current_layer + '_mnn'] = torch.mul(activation[current_layer + '_m'],
                                    activation[current_layer + '_up'](activation[last_layer + '_mnn']))    
        else:
            activation[current_layer + '_mpp'] = torch.mm(activation[current_layer + '_m'],
                                                           activation[last_layer + '_mpp'])
            activation[current_layer + '_mnn'] = torch.mm(activation[current_layer + '_m'],
                                                           activation[last_layer + '_mnn'])
        del (activation[current_layer + '_m'])

    else:
        if activation.get(current_layer + '_dx') is not None:  # conv
            sp = activation[current_layer + '_yshape']
            mpp = activation[last_layer + '_mpp'].reshape(sp[1], sp[2], sp[3], activation[last_layer + '_mpp'].shape[-1])
            mpp = mpp.permute(3, 0, 1, 2)
            activation[current_layer + '_mpp'] = activation[current_layer + '_pconv'](
                        mpp
                    ) * (activation[current_layer + '_dx'] > 0) + activation[current_layer + '_nconv'](
                        mpp
                    ) * (activation[current_layer + '_dx'] < 0) + activation[current_layer + '_tconv'](
                        mpp / 2
                    ) * (activation[current_layer + '_dx'] == 0)
            mnn = activation[last_layer + '_mnn'].reshape(sp[1], sp[2], sp[3],
                                                                  activation[last_layer + '_mnn'].shape[-1])
            mnn = mnn.permute(3, 0, 1, 2)
            activation[current_layer + '_mnn'] = activation[current_layer + '_pconv'](
                        mnn
                    ) * (activation[current_layer + '_dx'] < 0) + activation[current_layer + '_nconv'](
                        mnn
                    ) * (activation[current_layer + '_dx'] > 0) + activation[current_layer + '_tconv'](
                        mnn / 2
                    ) * (activation[current_layer + '_dx'] == 0)
            sp = activation[current_layer + '_mpp'].shape
            activation[current_layer + '_mpp'] = activation[current_layer + '_mpp'].reshape(
                        sp[0], sp[-1] * sp[-2] * sp[-3]).transpose(1, 0).to(device)
            activation[current_layer + '_mnn'] = activation[current_layer + '_mnn'].reshape(
                        sp[0], sp[-1] * sp[-2] * sp[-3]).transpose(1, 0).to(device)
            del activation[current_layer + '_dx']
            del activation[current_layer + '_pconv']
            del activation[current_layer + '_nconv']
            del activation[current_layer + '_tconv']
        else:
            activation[current_layer + '_mpp'] = torch.mm(activation[current_layer + '_mpp'],
                                                              activation[last_layer + '_mpp']).to(device)
            activation[current_layer + '_mnn'] = torch.mm(activation[current_layer + '_mnn'],
                                                              activation[last_layer + '_mnn']).to(device)
    #if last_layer != true_last_layer:
        #del (activation[last_layer + '_mpp'])
        #del (activation[last_layer + '_mnn'])
    return activation


def sample_Contribute(net, activation, x):
    with torch.no_grad():
        y = net(x)
        layerlist = list(net._modules)
        last_layer = layerlist.pop()
        while 'Softmax' in net._modules[last_layer]._get_name():
            last_layer = layerlist.pop()
        # activation_original = activation.copy()
        true_last_layer = last_layer
        while len(layerlist) != 0:
            current_layer = layerlist.pop()
            #print('current_layer: '+current_layer)
            if net._modules[current_layer]._get_name() == 'Sequential':
                sublayerlist = list(net._modules[current_layer]._modules)
                while len(sublayerlist) != 0:
                    current_sublayer = sublayerlist.pop()
                    #print('current_sublayer: '+current_sublayer)
                    activation = deal_layer_m(net._modules[current_layer],activation,last_layer,true_last_layer,current_sublayer)
                    last_layer = current_sublayer
            else:
                activation = deal_layer_m(net,activation,last_layer,true_last_layer,current_layer)
                last_layer = current_layer
    torch.cuda.empty_cache()
    gc.collect()
    return y, activation



