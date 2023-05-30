"""
The algorithm code is based on the github.com/utkuozbulak of author(Utku Ozbulak)
but we use the maximum value of the feature map channel to generate the attention map
"""

import os
import os.path as osp
import json
import torch
from torch.utils.data import dataset
import numpy as np
import torchvision
from torch.autograd import Variable
from torch.nn import ReLU

from function.ex_methods.module.func import save_gradient_images, load_image,loader2imagelist
from scipy.stats import kendalltau

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, model):
        self.model = self.process_model(model)
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_full_backward_hook(hook_function)

    def process_model(self, model):
        #set torch.nn.ReLU inplace to be False
        for name, module in model._modules.items():
            if isinstance(module, torch.nn.Sequential):
                for m in module:
                    if isinstance(m,torch.nn.ReLU):
                        m.inplace = False
            else:
                if isinstance(module, torch.nn.ReLU):
                    module.inplace = False
        return model

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
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_full_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def get_maxvalue_filter(self, x):
        x = torch.sum(x, dim=(2,3))
        index = torch.max(x,1)[1]
        return index

    def generate_gradients(self, input_image, cnn_layer):
        self.model.zero_grad()
        # Forward pass
        x = input_image
        for index, layer in enumerate(self.model.features):
            # Forward pass layer by layer
            # x is not used after this point because it is only needed to trigger
            # the forward hook function
            x = layer(x)
            # Only need to forward until the selected layer is reached
            if index == cnn_layer:
                # (forward hook function triggered)
                break
        filter_pos = self.get_maxvalue_filter(x)
        conv_output = torch.sum(torch.abs(x[0, filter_pos]))
        # Backward pass
        conv_output.backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

def get_model_and_layers(model_name):
    cnn_layers = None
    model = None
    
    if model_name == "vgg19":
        model = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        cnn_layers = [1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29, 31, 33, 35]
    elif model_name == "alexnet":
        model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        cnn_layers = [1, 4, 7, 9, 11]
    return model, cnn_layers

def layer_analysis(imgs_list, file_name_to_export, model_name, save_path):
    model, cnn_layers = get_model_and_layers(model_name)
    result = {}
    nor_ex_list = []
    adv_ex_list = []
    save_path = osp.join(save_path, model_name)
    # Guided backprop
    GBP = GuidedBackprop(model)
    for name in file_name_to_export:
        i = 0
        for prep_img in imgs_list.get(name):
            _prep_img = Variable(prep_img, requires_grad=True)
            # creat a folder
            if not os.path.exists(save_path+'/img_' + f"{i}/" + name):
                os.makedirs(save_path+'/img_'+f"{i}/" + name)
            if result.get(f'img_{i}') == None:
                result[f'img_{i}'] = {}
            if result.get("value") == None:
                result["value"] = {}
            tmp_list = []
            # [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
            for cnn_layer in cnn_layers:
                # File export name
                filename = save_path + '/img_' + \
                    str(i) + "/" + name + \
                    "/layer" + str(cnn_layer) + ".jpg"
                print(filename)
                # Get gradients
                guided_grads = GBP.generate_gradients(
                    _prep_img, cnn_layer)
                # Save colored gradients
                path = save_gradient_images(guided_grads, filename)
                tmp_list.append(guided_grads)
                result[f'img_{i}'].update({
                    f"{name}{cnn_layer}": path
                })
                # Convert to grayscale
                # grayscale_guided_grads = convert_to_grayscale(guided_grads)
                # Save grayscale gradients
                # save_gradient_images(grayscale_guided_grads, filename + '_Guided_BP_gray')
                # Positive and negative saliency maps
                # pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
                # save_gradient_images(pos_sal, filename + '_pos_sal')
                # save_gradient_images(neg_sal, filename + '_neg_sal')
            if name == "nor":
                nor_ex_list.append(tmp_list)
            else: 
                adv_ex_list.append(tmp_list)
            i = i + 1
    j = 0
    for nor_imgs, adv_imgs in zip(nor_ex_list,adv_ex_list):
        t_list = []
        for nor_layer, adv_layer in zip(nor_imgs,adv_imgs):    
            kendall_value, _ = kendalltau(np.array(nor_layer,dtype="float64"),np.array(adv_layer,dtype="float64"))
            t_list.append(kendall_value)
        result["value"].update({
            f"img_{j}": t_list
        })
        j = j + 1
    print('Layer Guided backprop completed')
    return result


def get_all_layer_analysis(model_name, nor_loader, adv_loader, dataset, save_path):
    device = torch.device("cpu")
    nor_tensor_imgs = []
    adv_tensor_imgs = []

    # model = model.to(device)
    file_to_export = ["nor", "adv"]
    loader = nor_loader
    nor_img_list = loader2imagelist(loader, dataset, size=5)
    for img in nor_img_list:
        nor_tensor_img = load_image(device, img, dataset)
        nor_tensor_imgs.append(nor_tensor_img)
    adv_loader = adv_loader
    adv_img_list = loader2imagelist(adv_loader, dataset,size=5)
    for img in adv_img_list:
        adv_tensor_img = load_image(device, img, dataset)
        adv_tensor_imgs.append(adv_tensor_img)
    list = {"nor": nor_tensor_imgs, "adv": adv_tensor_imgs}
    result = layer_analysis(list, file_to_export, model_name, save_path)
    return result
