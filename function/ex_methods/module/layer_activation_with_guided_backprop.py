"""
The algorithm code is based on the github.com/utkuozbulak of author(Utku Ozbulak)
but we use the maximum value of the feature map channel to generate the attention map
"""

import os
import os.path as osp
import json
import torch

import numpy as np
import torchvision
from torch.autograd import Variable
from .relu import ReLU

from function.ex_methods.module.func import save_gradient_images, load_image, loader2imagelist, get_conv_layer
from scipy.stats import kendalltau

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, model):
        self.model = model
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * \
                torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def get_maxvalue_filter(self, x):
        x = torch.sum(x, dim=(2,3))
        index = torch.max(x,1)[1]
        return index

    def generate_gradients(self, input_image, cnn_layer):
        self.model.zero_grad()
        # Forward pass
        x = input_image
        for index, layer in self.model._modules.items():
            # Forward pass layer by layer
            # x is not used after this point because it is only needed to trigger
            # the forward hook function
            x = layer(x)
            # Only need to forward until the selected layer is reached
            if index == cnn_layer:
                # (forward hook function triggered)
                break
        filter_pos = self.get_maxvalue_filter(x).item()
        conv_output = torch.sum(torch.abs(x[0, filter_pos]))
        # Backward pass
        # conv_output.backward()
        grad = torch.autograd.grad(conv_output, input_image,
                retain_graph=False, create_graph=False)[0]
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = grad.data.cpu().numpy()[0]
        return gradients_as_arr


def layer_analysis(model, imgs_list, model_name, save_path, adv_mehod):
    cnn_layers = get_conv_layer(model_name)
    result = {}
    ex_img_list = []
    # Guided backprop
    GBP = GuidedBackprop(model)

    for index, prep_img in enumerate(imgs_list):
        _prep_img = Variable(prep_img, requires_grad=True)
        # creat a folder
        if not os.path.exists(save_path+'/img_' + f"{index}" ):
            os.makedirs(save_path+'/img_'+f"{index}")
        if result.get(f'img_{index}') == None:
            result[f'img_{index}'] = {}
            result[f'img_{index}'][adv_mehod] = {}
        layer_list = []
        for cnn_layer in cnn_layers:
            # File export name
            filename = save_path + '/img_' + \
                str(index) + "/" + model_name + '_' + adv_mehod + \
                "_layer" + str(cnn_layer) + ".jpg"
            # Get gradients
            guided_grads = GBP.generate_gradients(
                _prep_img, cnn_layer)
            # Save colored gradients
            path = save_gradient_images(guided_grads, filename)
            layer_list.append(guided_grads)
            result[f'img_{index}'][adv_mehod].update({
                f"layer{cnn_layer}": path
            })
        ex_img_list.append(layer_list)

    return result, ex_img_list

