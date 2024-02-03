"""
These functions are used for reachability analysis of CNNs

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause

"""

import numpy as np
import itertools
import copy as cp
import torch
import torch.nn as nn
import torch.multiprocessing
from torch.autograd import Variable
import veritex.sets.cubelattice as cl
from function.ex_methods.module.func import Logger
import pickle, time, sys, os

class Flatten(nn.Module):
    """
    A class for the flatten layer in CNNs
    """
    def forward(self, input):
        return torch.flatten(input,1)

class Network:
    """
    A class for the CNN in the reachability analysis

        Attributes:
            flatten_pos (int): Layer index of the flatten layer
            is_cuda (bool): Enable cuda
            layer_gradients (list): Gradients of layers with respect to one output
            relaxation (float): Percentage of neurons that are accurately processed in the fast reachability analysis
            label (tensor): Class index of the image
            sequential (sequential): CNN model
            layer_num (int): Number of model layers
            layer_inputs (list): Input to each layer of the CNN with respect to the input image
            layer_process_range (list): Range of inputs that will be processed in each layer
            layer_attack_poss (list): Attack dimensions of inputs in each layer
            _layer (int): Index of the layer where computation is conducted
            _image_frame (np.ndarray): Block of the input that will be processed in each layer
            _slice_blocks (list): Dimensions of the input that will be processed in max pooling layers


    """
    def __init__(self, net, image, label, attack_pos, layer_gradients,
                 is_cuda=False, relaxation=1):

        """
        Construct attributes for a class object

        Parameters:
            net (pytorch): Network model
            image (tensor): Input image
            label (tensor): Class index of the image
            attack_pos (list): Attack positions of the image
            layer_gradients (list): Gradients of layers with respect to one output
            is_cuda (bool): Enable cuda
            relaxation (float): Percentage of neurons that are accurately processed in the fast reachability analysis
        """

        self.flatten_pos = None # the position of the flatten function in layers
        self.is_cuda = is_cuda
        self.layer_gradients = layer_gradients
        self.relaxation = relaxation
        self.label = label
        
        for param in net.parameters():
            param.requires_grad = False
        self.sequential = self.forward_layer_sequential(net)
        self.layer_num = len(self.sequential) # the number of layers
        self.layer_inputs = self.forward_layer_input(image)

        self.layer_process_range, self.layer_attack_poss = self.layers_range(attack_pos)
        # relaxation<1 indicates a fast reachability analysis
        if self.relaxation < 1:
            self.layer_selected_neurons = self.get_target_neurons()

        self._layer = None
        self._image_frame = None
        self._slice_blocks = None


    def forward_layer_sequential(self, net):
        """
        Convert CNN model to the sequential

        Parameters:
            net (pytorch): CNN model

        Returns:
            sequential_layers (sequential): Sequential model
        """
        net_layers = []

        for i in range(len(net.sequential)):
            type_name = type(net.sequential[i]).__name__

            if type_name in ['Flatten', 'MaxPool2d', 'BatchNorm2d','ReLU','DataParallel','Linear','Conv2d']:
                net_layers.append(net.sequential[i])
            if type_name == 'Flatten':
                self.flatten_pos = i

        if self.is_cuda:
            sequential_layers = nn.Sequential(*net_layers).eval().cuda()
        else:
            sequential_layers = nn.Sequential(*net_layers).eval()

        
        for param in sequential_layers.parameters():
            param.requires_grad = False

        return sequential_layers


    def forward_layer_input(self, image):
        """
        Compute the input to each layer of the CNN with respect to the input image

        Parameters:
            image (tensor): Input image to the CNN
            layer_inputs (list): Input to each layer of the CNN with respect to the input image
        """

        if self.is_cuda:
            im = cp.deepcopy(image).cuda()
            layer_inputs = [im.cpu().numpy()]
            for i in range(len(self.sequential)):
                im = self.sequential[i](im)
                layer_inputs.append(cp.deepcopy(im.cpu().numpy()))

        else:
            im = cp.deepcopy(image)
            layer_inputs = [im.numpy()]
            for i in range(len(self.sequential)):
                im = self.sequential[i](im)
                layer_inputs.append(cp.deepcopy(im.numpy()))

        return layer_inputs


    def regular_reach(self, input_image_fl):
        """
        Compute reachable sets of the CNN

        Parameters:
            input_image_fl (FlatticeCNN):

        Returns:
            layer_fls_out (list): Output reachable sets represented by their vertices
            and vertices of their corresponding subset in the input domain
        """
        layer_fls = input_image_fl
        for self._layer in range(self.layer_num):
            self.reach_single_layer(layer_fls)
        layer_fls_out = [[afl.vertices, afl.vertices_init] for afl in layer_fls]
        return layer_fls_out


    def index_convesion(self, indices, matrix_size):
        """
        Transform indices

        Parameters:
            indices (tensor): Indices to process
            matrix_size (size): Reference

        Returns:
            indices_new (np.ndarray): Transformed indices
        """
        raw = indices
        indices_new = []
        for adim in list(matrix_size)[::-1]:
            indices_new.append((raw % adim).numpy())
            raw = raw / adim

        indices_new = np.asarray(indices_new[::-1]).transpose()
        return indices_new


    def get_target_neurons(self):
        """
        Identify neurons that are accurately processed in the fast reachability analysis

        Returns:
            neurons_selected_layers (list): Selected neurons in each layer
        """
        neurons_selected_layers = []
        neurons_non_selected_layers = []
        for layer in range(self.layer_num-1): # the last layer is not considered
            layer_name = type(self.sequential[layer]).__name__
            if layer_name == 'Flatten':
                neurons_selected_layers.append([])
                neurons_non_selected_layers.append([])
                continue

            gradients = self.layer_gradients[layer]
            if layer < self.flatten_pos:
                pos = self.layer_attack_poss[layer]
                grads_pos = gradients[:, pos[0][0]:pos[1][0] + 1, pos[0][1]:pos[1][1] + 1]
                # Number of neurons to select
                tops = int(np.ceil(self.relaxation*grads_pos.shape[0]*grads_pos.shape[1]*grads_pos.shape[2]))
                grads_pos_faltten = grads_pos.flatten()
                _, neurons_topk = torch.topk(grads_pos_faltten, tops)

                neurons_new_topk = self.index_convesion(neurons_topk, grads_pos.size())

                if layer_name == 'MaxPool2d':
                    range_pos = self.layer_process_range[layer]
                    rel_pos = pos[0]-range_pos[0]
                    neurons_new_topk[:,1] += rel_pos[0]
                    neurons_new_topk[:,2] += rel_pos[1]
                    neurons_new_topk = [tuple(arr) for arr in neurons_new_topk]
                else:
                    neurons_new_topk = [tuple(arr) for arr in neurons_new_topk]

            else:
                grads_pos = gradients
                tops = int(np.ceil(self.relaxation *grads_pos.shape[0]))
                _, neurons_topk = torch.topk(grads_pos, tops)
                neurons_new_topk = neurons_topk.tolist()

            neurons_selected_layers.append(neurons_new_topk)

        return neurons_selected_layers


    def layers_range(self, initial_attack_pos):
        """
        Compute the attack positions and the process range of inputs in each layer

        Parameters:
            initial_attack_pos (list): Attack positions of the input images

        Returns:
            layer_range_pos (list): Process range of inputs in each layer
            layer_attack_pos (list): Attack positions of inputs in each layer
        """
        attack_pos = initial_attack_pos
        layer_range_pos = []
        layer_attack_pos = []
        for self._layer in range(self.flatten_pos):
            if type(self.sequential[self._layer]).__name__ == 'ReLU' or \
                    type(self.sequential[self._layer]).__name__ == 'BatchNorm2d':
                layer_attack_pos.append(attack_pos)
                layer_range_pos.append(attack_pos)
                continue

            layer_attack_pos.append(attack_pos)
            range_to_process, next_attack_pos = self.range_to_process_layer(attack_pos)
            layer_range_pos.append(range_to_process)
            attack_pos = next_attack_pos

        layer_attack_pos.append(attack_pos)
        return layer_range_pos, layer_attack_pos


    def reach_single_layer(self, all_fls):
        """
        Reachability analysis of one layer

        Parameters:
            all_fls (list): Input reachable sets and after computation it stores all output reachable sets
        """
        if type(self.sequential[self._layer]).__name__ == 'DataParallel':
            type_name = type(self.sequential[self._layer].module).__name__
        else:
            type_name = type(self.sequential[self._layer]).__name__
        if type_name == 'BatchNorm2d':
            self.norm2d_layer(all_fls)

        elif type_name == 'Conv2d':
            rp = self.layer_process_range[self._layer]
            self._image_frame = self.layer_inputs[self._layer][:, :, rp[0][0]:rp[1][0] + 1, rp[0][1]:rp[1][1] + 1]
            self.conv2d_layer(all_fls)

        elif type_name == 'ReLU' and self._layer<self.flatten_pos:
            neurons = np.array([])
            all_fls_len = len(all_fls)
            num = 0
            while num < all_fls_len:
                im_fl = all_fls[0]
                all_fls.pop(0)
                try: # Avoid numerical error
                    im_fl_output = self.relu_layer1(im_fl, neurons, False)
                    all_fls.extend(im_fl_output)
                except:
                    pass
                num += 1

        elif type_name == 'MaxPool2d':
            r2p = self.layer_process_range[self._layer]
            attack_pos = self.layer_attack_poss[self._layer]
            pos = attack_pos - r2p[0]
            self._slice_blocks = self.get_slice_blocks(r2p)
            self._image_frame = self.layer_inputs[self._layer][:, :, r2p[0][0]:r2p[1][0] + 1, r2p[0][1]:r2p[1][1] + 1]

            all_fls_length = len(all_fls)
            num = 0
            while num < all_fls_length:
                im_fl = all_fls[0]
                all_fls.pop(0)
                data_frame = self._image_frame.repeat(im_fl.vertices.shape[0], 0)
                data_frame[:, :, pos[0][0]:pos[1][0] + 1, pos[0][1]:pos[1][1] + 1] = cp.deepcopy(im_fl.vertices)
                im_fl.vertices = data_frame
                blocks = np.array([])
                try: # Avoid numerical errors
                    im_fl_output = self.maxpool2d_layer(im_fl, blocks, False)
                    all_fls.extend(im_fl_output)
                except:
                    pass
                num += 1

        elif type_name == 'Flatten':
            self._image_frame = self.layer_inputs[self._layer]

            for im_fl in all_fls:
                self.flatten_layer(im_fl)

        elif type_name == 'ReLU' and self._layer > self.flatten_pos:
            neurons = np.array([])
            all_fls_length = len(all_fls)
            num = 0
            while num < all_fls_length:
                im_fl = all_fls[0]
                all_fls.pop(0)
                try: # Avoid numerical errors
                    im_fl_output = self.relu_layer2(im_fl, neurons, False)
                    all_fls.extend(im_fl_output)
                except:
                    pass
                num += 1

        elif type_name == 'Linear':
            for im_fl in all_fls:
                self.linear_layer(im_fl)


    def norm2d_layer(self, all_fls):
        """
        Reachability analysis in BatchNorm2d layers

        Parameters:
            all_fls (list): Input reachable sets and after computation it stores all output reachable sets

        """
        if self.is_cuda:
            for im_fl in all_fls:
                vertices = cp.deepcopy(torch.from_numpy(im_fl.vertices).cuda())
                vertices_norm = self.sequential[self._layer](vertices)
                im_fl.vertices = vertices_norm.cpu().numpy()
        else:
            for im_fl in all_fls:
                vertices = torch.from_numpy(im_fl.vertices)
                vertices_norm = self.sequential[self._layer](vertices)
                im_fl.vertices = vertices_norm.numpy()


    def conv2d_layer(self, all_fls):
        """
        Reachability analysis in Conv2d layers

        Parameters:
            all_fls (list): Input reachable sets and after computation it stores all output reachable sets

        """
        if self.is_cuda:
            ap = self.layer_attack_poss[self._layer]
            ap_next = self.layer_attack_poss[self._layer+1]
            rp = self.layer_process_range[self._layer]
            for im_fl in all_fls:
                im_input = cp.deepcopy(self._image_frame).repeat(im_fl.vertices.shape[0], axis=0)
                ap2 = ap - rp[0]
                im_input[:, :, ap2[0][0]:ap2[1][0] + 1, ap2[0][1]:ap2[1][1] + 1] = cp.deepcopy(im_fl.vertices)
                # while self.check_gpu_and_block(im_input): {}

                im_input0 = torch.from_numpy(im_input).cuda()
                im_output0 = self.sequential[self._layer](im_input0)
                im_output = im_output0.cpu().numpy()

                torch.cuda.empty_cache()
                ap3 = ap_next - rp[0]
                im_fl.vertices = cp.deepcopy(im_output[:, :, ap3[0][0]:ap3[1][0] + 1, ap3[0][1]:ap3[1][1] + 1])
        else:
            ap = self.layer_attack_poss[self._layer]
            ap_next = self.layer_attack_poss[self._layer + 1]
            rp = self.layer_process_range[self._layer]
            for im_fl in all_fls:
                im_input = cp.deepcopy(self._image_frame).repeat(im_fl.vertices.shape[0], axis=0)
                ap2 = ap - rp[0]
                im_input[:, :, ap2[0][0]:ap2[1][0] + 1, ap2[0][1]:ap2[1][1] + 1] = im_fl.vertices
                im_input0 = torch.from_numpy(im_input)
                im_output0 = self.sequential[self._layer](im_input0)
                im_output = im_output0.numpy()
                ap3 = ap_next - rp[0]
                im_fl.vertices = cp.deepcopy(im_output[:, :, ap3[0][0]:ap3[1][0] + 1, ap3[0][1]:ap3[1][1] + 1])


    def get_valid_neurons1(self, afl, neurons):
        """
        Compute ReLU neurons whose input ranges are both spanned by the input set. It is before the flatten layer

        Parameters:
            afl (FlatticeCNN): Input set
            neurons (np.ndarray): Indices of neuron candidates

        Returns:
            valid_neurons_neg_pos (np.ndarray): Indices of neurons whose positive and negative input ranges are both spanned by the input set
            valid_neurons_neg (np.ndarray): Indices of neurons whose only negative input range is spanned by the input set
        """
        if neurons.shape[0] ==0:
            flag_neg = (afl.vertices<=0)
            temp_neg = np.all(flag_neg, 0)
            valid_neurons_neg = np.asarray(np.nonzero(temp_neg)).T
            temp_pos = np.all(afl.vertices>=0, 0)
            neurons_sum = temp_neg+temp_pos
            valid_neurons_neg_pos = np.asarray(np.nonzero(neurons_sum==False)).T
            return valid_neurons_neg_pos, valid_neurons_neg

        elements = afl.vertices[:, neurons[:,0], neurons[:,1], neurons[:,2]]
        flag_neg = (elements <= 0)
        temp_neg = np.all(flag_neg, 0)
        temp_pos = np.all(elements>=0, 0)
        temp_sum = temp_neg + temp_pos
        indx_neg_pos = np.asarray(np.nonzero(temp_sum == False)).T
        valid_neurons_neg_pos = neurons[indx_neg_pos[:,0]]
        indx_neg = np.asarray(np.nonzero(temp_neg)).T
        valid_neurons_neg = neurons[indx_neg[:,0]]

        return valid_neurons_neg_pos, valid_neurons_neg


    def get_valid_neurons2(self, afl, neurons):
        """
        Compute ReLU neurons whose input ranges are both spanned by the input set. It is after the flatten layer

        Parameters:
            afl (FlatticeCNN): Input set
            neurons (np.ndarray): Indices of neuron candidates

        Returns:
            valid_neurons_neg_pos (np.ndarray): Indices of neurons whose positive and negative input ranges are both spanned by the input set
            valid_neurons_neg (np.ndarray): Indices of neurons whose only negative input range is spanned by the input set
        """
        if neurons.shape[0] ==0:
            flag_neg = (afl.vertices<=0)
            temp_neg = np.all(flag_neg, 0)
            valid_neurons_neg = np.asarray(np.nonzero(temp_neg)).T
            temp_pos = np.all(afl.vertices>=0, 0)
            neurons_sum = temp_neg + temp_pos
            valid_neurons_neg_pos = np.asarray(np.nonzero(neurons_sum==False))[0,:]
            return valid_neurons_neg_pos, valid_neurons_neg

        elements = afl.vertices[:,neurons[:,0]]
        flag_neg = (elements <= 0)
        temp_neg = np.all(flag_neg, 0)
        valid_neurons_neg = np.asarray(np.nonzero(temp_neg)).T
        temp_pos = np.all(elements>=0, 0)
        neurons_sum = temp_neg + temp_pos
        indx_neg_pos = np.asarray(np.nonzero(neurons_sum==False)).T
        valid_neurons_neg_pos = neurons[indx_neg_pos[:, 0]][0,:]
        return valid_neurons_neg_pos, valid_neurons_neg


    def relu_layer1(self, im_fl, neurons, flag=True):
        """
        Reachability analysis in ReLU layers. It is before the flatten layer

        Parameters:
            im_fl (FlatticeCNN): Input reachable set
            neurons (np.ndarray): Indices of neurons to process
            flag (bool): Help to determine the end of computation

        Returns:
            all_fls (list): All output reachable sets
        """
        if (neurons.shape[0] == 0) and flag:
            return [im_fl]

        new_neurons, new_neurons_neg = self.get_valid_neurons1(im_fl, neurons)

        im_fl.map_negative_fl_multi_relu1(new_neurons_neg)

        if new_neurons.shape[0] == 0:
            return [im_fl]

        if self.relaxation<1 and (tuple(new_neurons[0]) not in self.layer_selected_neurons[self._layer]):
            im_fl.fast_reach = True # Enable fast reachability analysis
        fls = self.split_facelattice(im_fl, new_neurons[0], 'relu1')

        new_neurons = new_neurons[1:]

        all_fls = []
        for afl in fls:
            all_fls.extend(self.relu_layer1(afl, new_neurons))

        return all_fls


    def relu_layer2(self, im_fl, neurons, flag=True):
        """
        Reachability analysis in ReLU layers. It is after the flatten layer

        Parameters:
            im_fl (FlatticeCNN): Input reachable set
            neurons (np.ndarray): Indices of neurons to process
            flag (bool): Help to determine the end of computation

        Returns:
            all_fls (list): All output reachable sets
        """
        if (neurons.shape[0] == 0) and flag:
            return [im_fl]

        new_neurons, new_neurons_neg = self.get_valid_neurons2(im_fl, neurons)
        im_fl.map_negative_fl_multi_relu2(new_neurons_neg)

        if new_neurons.shape[0] == 0:
            return [im_fl]

        if self.relaxation<1 and (new_neurons[0] not in self.layer_selected_neurons[self._layer]):
            im_fl.fast_reach = True # Enable fast reachability analysis
        fls = self.split_facelattice(im_fl, new_neurons[0], 'relu2')
        new_neurons = new_neurons[1:]

        all_fls = []
        for afl in fls:
            all_fls.extend(self.relu_layer2(afl, new_neurons))

        return all_fls


    def linear_layer(self, im_fl):
        """
        Affine Mapping between layers

        Parameters:
            im_fl (FlatticeCNN): Input set and it becomes the output reachable set after the computation
        """
        if self.is_cuda:
            im_input = im_fl.vertices
            # while self.check_gpu_and_block(im_input): pass
            temp0 = torch.from_numpy(im_input).cuda()

            temp1 = self.sequential[self._layer](temp0)
            im_fl.vertices = temp1.cpu().numpy()
            torch.cuda.empty_cache()

        else:
            temp0 = torch.from_numpy(im_fl.vertices)
            temp1 = self.sequential[self._layer](temp0)
            im_fl.vertices = temp1.numpy()


    def split_facelattice(self, im_fl, aneuron, split_type):
        """
        Conduct split operation on a reachable set w.r.t. ReLU function in ReLU or Maxpool layers

        Parameters:
            im_fl (FlatticeCNN): Input set
            aneuron (np.ndarray): Index of the ReLU neuron or function
            split_type (str): Name of the layer

        Outputs:
            outputs (list): Output reachable sets
        """
        if split_type=='relu1': # ReLU layers before the Faltten layer
            pos_fl, neg_fl = im_fl.single_split_relu1(aneuron)
        elif split_type=='maxpool':
            pos_fl, neg_fl = im_fl.single_split_maxpool(aneuron)
        elif split_type =='relu2': # ReLU layers before the Faltten layer
            pos_fl, neg_fl = im_fl.single_split_relu2(aneuron)
        else:
            sys.exit('Split type is not defined!\n')

        outputs = []
        if pos_fl:
            outputs.append(pos_fl)
        if neg_fl:
            outputs.append(neg_fl)

        return outputs


    def get_valid_blocks(self, blocks_tosplit, indices):
        """
        Identify new blocks in Maxpool layers where an input set will be split

        Parameters:
            blocks_tosplit (np.ndarray): Block candidates
            indices (np.ndarray): Indices of dimensions that are maximal in the max pooling

        Returns:
            valid_blocks (np.ndarray): New blocks
        """
        if blocks_tosplit.shape[0]==0:
            flag_equ = (indices==indices[0,:,:,:])
            valid_blocks = np.asarray(np.nonzero(np.any(flag_equ==False, 0))).T
            return valid_blocks

        valid_indices = indices[:,blocks_tosplit[:,0],blocks_tosplit[:,1],blocks_tosplit[:,2]]
        flag_equal = (valid_indices==valid_indices[0,:])
        indx_temp = np.asarray(np.nonzero(np.any(flag_equal == False, 0))).T
        valid_blocks = blocks_tosplit[indx_temp[:,0]]
        return valid_blocks


    def maxpool2d_layer(self, image_fl, blocks_tosplit, flag=True):
        """
        Reachability analysis in Maxpool layers

        Parameters:
            image_fl (FlatticeCNN): Input set
            blocks_tosplit (np.ndarray): Blocks where input sets are split
            flag (bool): Help to determine the end of computation

        Returns:
            all_fls (list): All output reachable sets
        """

        if self.is_cuda:
            self.sequential[self._layer].return_indices = True
            temp0 = torch.from_numpy(image_fl.vertices).cuda()
            layer_outs0, indices0 = self.sequential[self._layer](temp0)
            self.sequential[self._layer].return_indices = False
            layer_outs = layer_outs0.cpu().numpy()
            indices = indices0.cpu().numpy()
            torch.cuda.empty_cache()
        else:
            self.sequential[self._layer].return_indices = True
            temp0 = torch.from_numpy(image_fl.vertices)
            layer_outs0, indices0 = self.sequential[self._layer](temp0)
            self.sequential[self._layer].return_indices = False
            layer_outs = layer_outs0.numpy()
            indices = indices0.numpy()

        if blocks_tosplit.shape[0]==0 and flag:
            image_fl.vertices = layer_outs
            return [image_fl]

        # Identify blocks where input sets are split
        blocks_tosplit_new = self.get_valid_blocks(blocks_tosplit, indices)

        if blocks_tosplit_new.shape[0]==0:
            image_fl.vertices = layer_outs
            return [image_fl]

        # Derive hyperplanes that will split input sets
        ablock = blocks_tosplit_new[0]
        blocks_tosplit_new = blocks_tosplit_new[1:]
        indices_flatten = np.unique(indices[:,ablock[0],ablock[1],ablock[2]])
        aset_elements = [[ablock[0]]+self._slice_blocks[idx] for idx in indices_flatten]
        hyperplanes = list(itertools.combinations(aset_elements, 2))

        # Conduct the split operation using each hyperplane
        fls_temp = [image_fl]
        for hp in hyperplanes:
            fls_temp_hp = []
            if self.relaxation<1 and (tuple(hp[0]) not in self.layer_selected_neurons[self._layer]) and \
                    (tuple(hp[1]) not in self.layer_selected_neurons[self._layer]):
                temp_fast = True
            else:
                temp_fast = False

            for afl in fls_temp:
                afl.fast_reach = temp_fast
                fls_temp_hp.extend(self.split_facelattice(afl, hp, 'maxpool'))

            fls_temp = fls_temp_hp

        all_fls = []
        for afl in fls_temp:
            all_fls.extend(self.maxpool2d_layer(afl, blocks_tosplit_new))

        return all_fls


    def get_slice_blocks(self, range_to_process):
        """
        Compute dimensions of the input that will be processed in reachability analysis of max pooling layers

        Parameters:
            range_to_process (list): Dimension range to process

        Returns:
            blocks (list): All dimensions of the input to process
        """
        width = range_to_process[1][1]-range_to_process[0][1]+1
        height = range_to_process[1][0]-range_to_process[0][0]+1
        blocks = []
        for h in range(height):
            for w in range(width):
                ablock = [h, w]
                blocks.append(ablock)

        return blocks


    def flatten_layer(self, image_fl):
        """
        Reachability analysis in Flatten layer

        Parameters:
            image_fl (FlatticeCNN): Input set and after the computation it becomes the output set
        """
        ap = self.layer_attack_poss[self._layer]
        data_frame = cp.deepcopy(self._image_frame).repeat(image_fl.vertices.shape[0], 0)
        data_frame[:, :, ap[0][0]:ap[1][0] + 1, ap[0][1]:ap[1][1] + 1] = cp.deepcopy(image_fl.vertices)
        image_fl.vertices = data_frame.reshape(image_fl.vertices.shape[0],-1)


    def range_to_process_layer(self, attack_pos):
        """
        Compute the attack positions and the process range of inputs in one layer

        Parameters:
            attack_pos (list): Attack positions of the input images in the current layer

        Returns:
            range_to_process (list): Process range of inputs in the current layer
            next_attack_pos (list): Attack positions of inputs in the next layer
        """
        if type(self.sequential[self._layer]).__name__ == 'DataParallel':
            layer_function = self.sequential[self._layer].module
        else:
            layer_function = self.sequential[self._layer]

        kernel_size = np.array(layer_function.kernel_size)
        stride = np.array(layer_function.stride)
        padding = np.array(layer_function.padding)
        ub_pos = attack_pos[1]
        lb_pos = attack_pos[0]
        nmax = np.floor(np.divide(ub_pos + padding, stride)).astype(int)
        nmin = np.ceil(np.divide(lb_pos + padding + 1 - kernel_size, stride)).astype(int)
        nmin[nmin<0] = 0
        if nmax[0] > self.layer_inputs[self._layer].shape[2] - 1:
            nmax[0] = self.layer_inputs[self._layer].shape[2] - 1
        if nmax[1] > self.layer_inputs[self._layer].shape[3] - 1:
            nmax[1] = self.layer_inputs[self._layer].shape[3] - 1

        lb = nmin * stride - padding
        lb[lb < 0] = 0
        ub = nmax * stride + kernel_size - padding - 1
        if ub[0] > self.layer_inputs[self._layer].shape[2] - 1:
            ub[0] = self.layer_inputs[self._layer].shape[2] - 1
        if ub[1] > self.layer_inputs[self._layer].shape[3] - 1:
            ub[1] = self.layer_inputs[self._layer].shape[3] - 1

        range_to_process = [lb, ub]
        next_attack_pos = [nmin, nmax]

        return range_to_process, next_attack_pos


class Hook():
    """
    A class for the computation of gradients in CNNs
    """
    def __init__(self, module=None, backward=False):
        self.input = None
        self.output = None
        if module != None:
            if backward == False:
                self.hook = module.register_forward_hook(self.hook_fn)
            else:
                self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class Method:
    """
    A class for reachability analysis methods

        Attributes:
            model (pytorch): Network model
            image_orig (tensor): Original input image
            image (tensor): Image processed in layers
            label (tensor): Class index of the input image
            attack_block (tuple): Size of the attack block in the input image
            attack_target (int): Adversarial label to misclassify to, None if not specified
            epsilon (float): Perturbation epsilon
            num_core (int): Number of processors for computation
            is_cuda (bool): Enable cuda
            mean (np.ndarray): Mean for normalization of the input image
            std (np.ndarray): Std for normalization of the input image
            relaxation (float): Percentage of neurons that are accurately processed in the fast reachability analysis
            attack_poss_ranks (list): pixels to attack that are ranked in terms of their sensitivity (gradients)
            layer_gradients (list): Gradients of layers with respect to one output
            attack_poss (list): the pixel that has the largest gradient
            elapsed_time (float): Running time
            savepath (str): Path to save results

    """
    def __init__(self, model, image, label, file_path,
                 attack_block=(1,1),
                 epsilon=0.001,
                 relaxation=1,
                 target=None,
                 mean=np.array([0,0,0]),
                 std=np.array([1,1,1]),
                 is_cuda=False):

        """
        Construct attributes for a class object

        Parameters:
            model (pytorch): Network model
            image (tensor): Image processed in layers
            label (tensor): Class index of the input image
            file_path (str): Path to save results
            attack_block (tuple): Size of the attack block in the input image
            epsilon (float): Perturbation epsilon
            relaxation (float): Percentage of neurons that are accurately processed in the fast reachability analysis
            target (int): Adversarial label to misclassify to, None if not specified
            mean (np.ndarray): Mean for normalization of the input image
            std (np.ndarray): Std for normalization of the input image
            is_cuda (bool): Enable cuda
        """
        self.model = model
        self.image_orig = cp.deepcopy(image)
        self.image = image
        self.label = label
        self.attack_block = attack_block
        self.attack_target = target
        self.epsilon = epsilon
        self.num_core = torch.multiprocessing.cpu_count()-1
        self.is_cuda = is_cuda

        self.mean, self.std = mean, std
        self.attack_poss_ranks, self.layer_gradients = self.get_layer_gradients()
        self.attack_poss = self.attack_poss_ranks[0]

        self.elapsed_time = 0
        if not os.path.isdir(file_path):
            os.mkdir(file_path)
        self.savepath = file_path
        self.relaxation = relaxation
        if self.attack_target == self.label:
            sys.exit('Label should not be the attack target')


    def reach(self):
        """
        Compute output reachable sets of the CNN for the pixel perturbation

        Returns:
            all_fls (list): Output reachable sets
        """
        t0 = time.time()
        net = Network(self.model, self.image, self.label, self.attack_poss,
                           self.layer_gradients, relaxation=self.relaxation, is_cuda=self.is_cuda)

        self.attack_range = self.attack_range_3channel()
        # Partition input set into subsets
        all_input_fls = cl.partition_input(self.attack_range, pnum=4, poss=self.attack_poss)
        # pool = torch.multiprocessing.Pool(self.num_core)  # multiprocessing
        # pool = torch.multiprocessing.Pool(1)
        outputSets = []
        outputSets.append(net.regular_reach(all_input_fls))
        # outputSets.extend(pool.imap(net.regular_reach, all_input_fls))
        # pool.close()
        self.elapsed_time = time.time() - t0
        all_fls = [item for sublist in outputSets for item in sublist]

        filename = f'image_label_{self.label.numpy()}_epsilon' \
                   f'_{self.epsilon}_relaxation_{self.relaxation}'
        with open(self.savepath+'/'+filename+'.pkl', 'wb') as f:
            pickle.dump({'Output reachable sets':all_fls,
                         'Image label':self.label.numpy(),
                         'Running time':self.elapsed_time,
                         'Attack pixels':self.attack_poss,
                         'Relaxtion factor':self.relaxation}, f)
        return all_fls


    def attack_range_3channel(self):
        """
        Compute lower bounds and upper bounds of the input set from the pixel perturbation

        Returns:
            lb (np.ndarray): Lower bounds
            up (np.ndarray): Upper bounds
        """
        blocks = self.image[:,:,self.attack_poss[0][0]:(self.attack_poss[1][0]+1), self.attack_poss[0][1]:(self.attack_poss[1][1]+1)]

        for n in range(3):
            blocks[0,n,:,:] = blocks[0,n,:,:]*self.std[n]+self.mean[n]

        ub = cp.copy(blocks + self.epsilon)
        lb = cp.copy(blocks - self.epsilon)
        ub[ub>1.0] = 1.0
        lb[lb<0.0] = 0.0

        for n in range(3):
            ub[0,n,:,:] = (ub[0,n,:,:]-self.mean[n])/self.std[n]
            lb[0,n,:,:] = (lb[0,n,:,:]-self.mean[n])/self.std[n]

        return [lb[0].numpy(), ub[0].numpy()]


    def simulate(self, num=10000):
        """
        Compute outputs for images randomly generated from the perturbation

        Parameters:
            num (int): Number of simulations

        Returns:
            outputs (np.ndarray): Outputs of simulations
        """
        self.attack_range = self.attack_range_3channel()
        lbs = self.attack_range[0].flatten()
        ubs = self.attack_range[1].flatten()

        rands = []
        for idx in range(len(lbs)):
            lb = lbs[idx]
            ub = ubs[idx]
            rands.append(np.random.uniform(lb, ub, num))

        rands = np.array(rands).transpose().reshape((num, 3, self.attack_block[0], self.attack_block[1]))

        image = self.image_orig.numpy()
        ap = self.attack_poss
        outputs = np.array([])
        for n in range(num//100):
            sub = rands[n*100:(n+1)*100]
            image_frame = image.repeat(sub.shape[0], 0)
            image_frame[:, :, ap[0][0]: ap[1][0] + 1, ap[0][1]: ap[1][1] + 1] = sub
            image_frame = torch.tensor(image_frame)
            if n==0:
                outputs = self.model(image_frame).numpy()
            else:
                outputs = np.concatenate((outputs, self.model(image_frame).numpy()), axis=0)

        return outputs


    def get_layer_gradients(self):
        """
        Rank pixels in terms of their sensitivity and compute gradients of layers with respect to one output

        Returns:
            attack_poss_ranks (list): Pixels to attack that are ranked in terms of their sensitivity (gradients)
            layer_gradients (list): Gradients of layers with respect to one output
        """

        # create hooks for computation of gradients
        hook_back = []

        for layer in list(self.model._modules.items()):
            if layer[1]._modules != {}:
                for sub_layer in list(layer[1]._modules.items()):
                    if (type(sub_layer[1]).__name__ != 'Dropout2d') and \
                            (type(sub_layer[1]).__name__ != 'Dropout'):
                        hook_back.append(Hook(sub_layer[1], backward=True))
            else:
                hook_back.append(Hook(layer[1], backward=True))

            hook_back.append(Hook())  # for the flatten layer

        hook_back.pop(-1)
        x = Variable(self.image, requires_grad=True)
        y = self.model.forward(x)
        if self.label != y.argmax(axis=-1):
            print('This shouldn\' happen!')
            self.label = y.argmax(axis=-1)

        if self.attack_target == None:
            label_temp = self.label
        else:
            label_temp = self.attack_target

        y[0, label_temp].backward(retain_graph=True)

        layer_gradients = []
        for layer in hook_back:
            if layer.input != None:
                for ele in layer.input:
                    if ele != None:
                        layer_gradients.append(torch.squeeze(ele))
                        break
            else:
                layer_gradients.append([])  # for the flatten layer

        image_grad = torch.squeeze(hook_back[0].input[0])

        im_h = self.image.shape[2]
        im_w = self.image.shape[3]
        image_abs_grads = []
        for i in range(im_h - self.attack_block[0]):
            for j in range(im_w-self.attack_block[1]):
                tb_grad = image_grad[:,i:(i+self.attack_block[0]),j:(j+self.attack_block[1])]
                abs_grads = torch.sum(torch.abs(tb_grad))
                image_abs_grads.append(abs_grads)

        image_abs_grads = torch.tensor(image_abs_grads)
        idx_sorted = image_abs_grads.argsort(descending=True)

        # Rank pixel blocks in terms of their sensitivity
        attack_poss_ranks = []
        for idx in idx_sorted:
            hb = torch.div(idx,(im_h - self.attack_block[0]), rounding_mode='trunc')
            wb = idx%(im_w-self.attack_block[1])
            attack_poss_ranks.append([np.array([hb, wb]),
                                     np.array([hb+self.attack_block[0]-1, wb+self.attack_block[1]-1])])

        return attack_poss_ranks, layer_gradients