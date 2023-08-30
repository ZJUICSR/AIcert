"""
These functions are used for reachability analysis of feed-forward neural networks

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause

"""

import sys
import numpy as np
import copy as cp
from veritex.sets.vzono import VzonoFFNN as Vzono
from veritex.networks.funcs import relu, sigmoid, tanh
import torch.nn as nn
import torch




class FFNN:
    """
    A class to construct a network model and conduct reachability analysis

    Attributes:
        _W (list): Weight matrix between the network layers
        _b (list): Bias vector between the network layers
        _f (list): Activation functions in each layer
        _num_layer (int): The number of layers in the network model
        property (Property): One safety property
        verification (bool): Enable the safety verification of the network
        linearization (bool): Enable the linearization of activation functions
        unsafe_inputd (bool): Enable the computation of the entire unsafe input domain.
        exact_outputd (bool): Enable the computation of the exact unsafe output domain.
        repair (bool): Enable the repair of the network.

    Methods:
        set_property(safety_property):
            Set the safety property of the network.
        extract_params(torch_model):
            Extract parameters such as weights, bias and activation functions from a torch model.
        backtrack(s):
            Backtrack the unsafe input subspace with respect to an output reachable set.
        reach_over_approximation(s):
            Over approximate the output reachable domain of the network given an input set.
        layer_over_approximation(s, l):
            Over approximate the outpt reachable domain of one layer given its input set.
        reach_over_tuple(state_tuple):
            Over approximate the output reachable domain of the nework given a state tuple.
        compute_state(tuple_state):
            Compute the next state tuples given a state tuple
        verify(s):
            Verify the safey of a reachable set in FVIM or Flattice on the safety property.
        verify_vzono(s):
            Verify the safey of a reachable set in Vzono on the safety property.
    """

    def __init__(self, model,
                 repair=False,
                 verification=False,
                 linearization=False,
                 unsafe_inputd=False,
                 exact_outputd=False,
                 safety_property=None):
        """
        Constructs all the necessary attributes for the network object

            Parameters:
                model (list or pytorch): Network model
                repair (bool): Enable the repair function
                verification (bool): Enable the verification function
                linearization (bool): Enable the linearization of activation functions
                unsafe_inputd (bool): Enable the computation of the entire unsafe input domain
                exact_outputd (bool): Enable the computation of the exact output reachable domain
                safety_property (Property): A safety property

        """

        if isinstance(model, torch.nn.Sequential):
            self.extract_params(model)
        else:
            self._W, self._b, self._f = model

        # The last layer may not have activation functions
        assert len(self._f)==len(self._W) or len(self._f)+1==len(self._W)
        assert len(self._W)==len(self._b)
        self._num_layer = len(self._W)

        # safety properties
        self.property = safety_property

        # configurations for reachability analysis
        self.verification = verification
        self.linearization = linearization
        self.unsafe_inputd = unsafe_inputd
        self.exact_outputd = exact_outputd
        self.repair = repair

        # relu linearization does not support computation of unsafe input domains or exact output domains
        assert not(self.exact_outputd and self.linearization)
        assert not (self.unsafe_inputd and self.linearization)


    def set_property(self, safety_property):
        """
        Set the safety property of the network

            Parameters:
                safety_property (Property): One safety property

        """

        self.property = safety_property


    def extract_params(self, torch_model):
        """
        Extract the parameters from a pytorch model

            Parameters:
                torch_model (Pytorch): Pytorch model
        """

        self._W = []
        self._b = []
        # Extract the weights and bias between layers
        for name, param in torch_model.named_parameters():
            if name[-4:] == 'ight':
                if torch.cuda.is_available():
                    self._W.append(cp.deepcopy(param.data.cpu().numpy()))
                else:
                    self._W.append(cp.deepcopy(param.data.numpy()))
            if name[-4:] == 'bias':
                if torch.cuda.is_available():
                    temp = np.expand_dims(cp.deepcopy(param.data.cpu().numpy()), axis=1)
                    self._b.append(temp)
                else:
                    temp = np.expand_dims(cp.deepcopy(param.data.numpy()), axis=1)
                    self._b.append(temp)

        # Extract the name of activation functions in layers
        self._f = []
        for layer in torch_model:
            if isinstance(layer, nn.ReLU):
                self._f.append('ReLU')
            elif isinstance(layer, nn.Sigmoid):
                self._f.append('Sigmoid')
            elif isinstance(layer, nn.Tanh):
                self._f.append('Tanh')


    def backtrack(self, s):
        """
        Backtrack the input subspace given an output reachable set

            Parameters:
                s (FVIM or Flattice): An output reachable set

            Returns:
                inputs (FVIM or Flattice): A set of unsafe input sets

        """

        inputs = []
        for i in range(len(self.property.unsafe_domains)):
            As_unsafe = self.property.unsafe_domains[i][0]
            ds_unsafe = self.property.unsafe_domains[i][1]
            elements = np.dot(np.dot(As_unsafe,s.M), s.vertices.T) + np.dot(As_unsafe, s.b) +ds_unsafe
            if np.any(np.all(elements>0, axis=1)): # reachable set does not satisfy at least one linear constraint
                continue

            unsafe_s = cp.deepcopy(s)
            for j in range(len(As_unsafe)):
                A = As_unsafe[[j]]
                d = ds_unsafe[[j]]
                sub0 = unsafe_s.relu_split_hyperplane(A, d)
                if sub0:
                    unsafe_s = sub0
                else:
                    unsafe_s = []
                    break

            if unsafe_s: # in FVIM or Flattice
                inputs.append(unsafe_s)

        return inputs


    def reach_over_approximation(self, s=None):
        """
        Over approximate the output reachable domain of the network given an input set

            Parameters:
                s (Vzono): An input set to the network

            Returns:
                s (Vzono): An output reachable domain of the network
        """
        if s is None:
            s = self.property.input_set

        for l in range(self._num_layer):
            s = self.layer_over_approximation(s, l)
        return s


    def simulate(self, inputs=None, num=1000):
        """
        Comput outputs for input points

        Parameters:
            inputs (np.ndarray): Input points
            num (int): Number of inputs generated if 'inputs' is none

        Returns:
            inputs (np.ndarray): Output values
        """
        if inputs is None:
            lbs, ubs = self.property.lbs, self.property.ubs
            inputs = []
            for i in range(len(lbs)):
                inputs.append(torch.rand(num)*(ubs-lbs)+lbs)

        inputs = torch.tensor(inputs)
        for layer in range(self._num_layer):
            inputs = torch.matmul(torch.tensor(self._W[layer]),inputs) + torch.tensor(self._b[layer])
            if layer <= len(self._f)-1:
                if self._f[layer] == 'ReLU':
                    f = torch.nn.ReLU()
                    inputs = f(inputs)
                elif self._f[layer] == 'Sigmoid':
                    f = torch.nn.Sigmoid()
                    inputs = f(inputs)
                elif self._f[layer] == 'Tanh':
                    f = torch.nn.Tanh()
                    inputs = f(inputs)
        return inputs.numpy()


    def layer_over_approximation(self, s, l):
        """
        Over approximate the output reachable domain of one layer given its input set

            Parameters:
                s (Vzono): An input set
                l (int): Index of the layer

            Returns:
                s (Vzono): An output reachable set
        """

        # Affine mapping
        s.affine_map(self._W[l], self._b[l])

        # The last layer may not other activation functions
        if l <= len(self._f)-1:
            if self._f[l] == 'ReLU':
                s = relu.layer_linearize(s)
            elif self._f[l] == 'Sigmoid':
                s = sigmoid.layer_linearize(s)
            elif self._f[l] == 'Tanh':
                s = tanh.layer_linearize(s)
            else:
                sys.exit(f'{self._f[l]} is not supported!')

        return s


    def reach_over_tuple(self, state_tuple):
        """
        Over approximate the output reachable domain of the nework given a state tuple.

            Parameters:
                state_tuple (tuple): (a reachable set, index of the layer, neurons to process)

            Returns:
                vzono_set (Vzono): An output reachable set of the network
        """

        # Convert the set into Vzono
        s, layer, neurons = state_tuple
        base_vertices = np.dot(s.M, s.vertices.T) + s.b
        base_vectors = np.zeros((base_vertices.shape[0], 1))
        vzono_set = Vzono(base_vertices, base_vectors)

        neurons_neg_pos, neurons_neg = vzono_set.get_valid_neurons_for_over_app()
        vzono_set.base_vertices[neurons_neg,:] = 0
        vzono_set.base_vectors[neurons_neg,:] = 0

        for n in range(layer+1, self._num_layer):
            vzono_set = self.layer_over_approximation(vzono_set, n)

        return vzono_set


    def verify_vzono(self, s):
        """
        Check if the reachable set overlaps with the unsafe output domains

            Parameters:
                s (Vzono): An output reachable domain

            Returns:
                safe (bool): Whether the safety of the reachable set is safe or unknown
        """
        safe = []
        for indx, ud in enumerate(self.property.unsafe_domains):
            As_unsafe = ud[0]
            ds_unsafe = ud[1]
            safe.append(False)
            for n in range(len(As_unsafe)):
                A = As_unsafe[[n]]
                d = ds_unsafe[[n]]
                base_vertices = np.dot(A, s.base_vertices) + d
                base_vectors = np.dot(A, s.base_vectors)
                vals = base_vertices - np.sum(np.abs(base_vectors),axis=1)
                if np.all(vals>0):
                    safe[indx] = True
                    break

            if not safe[indx]: break

        return np.all(safe)


    def verify(self, s):
        """
        Check if the reachable set overlaps with the unsafe output domains.

            Parameters:
                s (FVIM or Flattice): An output reachable set

            Returns:
                unsafe (bool): Whether the safety of the reachable set is safe or unsafe
        """

        def verify_set(s, ud=None):
            As_unsafe = ud[0]
            ds_unsafe = ud[1]
            elements = np.dot(np.dot(As_unsafe, s.M), s.vertices.T) + np.dot(As_unsafe, s.b) + ds_unsafe
            if np.any(
                    np.all(elements >= 0, axis=1)):  # reachable set does not satisfy at least one linear constraint
                return False
            if np.any(np.all(elements <= 0, axis=0)):  # at least one vertex locates in unsafe domain
                return True
            unsafe_s = cp.deepcopy(s)
            for j in range(len(As_unsafe)):
                A = As_unsafe[[j]]
                d = ds_unsafe[[j]]
                sub0 = unsafe_s.relu_split_hyperplane(A, d)
                if sub0:
                    unsafe_s = sub0
                else:
                    return False  # vfl_set does not contain any unsafe elements
            return True  # unsafe_vfl is not none and contains unsafe elements

        unsafe = False
        for ud in self.property.unsafe_domains:
            A_unsafe = ud[0]
            d_unsafe = ud[1]
            if len(A_unsafe) == 1:
                vertices = np.dot(s.vertices, s.M.T) + s.b.T
                vals = np.dot(A_unsafe, vertices.T) + d_unsafe
                if np.any(np.all(vals<=0, axis=0)):
                    unsafe = True
                    break
            else:
                unsafe = verify_set(s, ud=ud)
                if unsafe:
                    break

        return unsafe


    def compute_state(self, tuple_state):
        """
        Compute the next state tuples given a state tuple

            Parameters:
                state_tuple (tuple): (a reachable set, index of the layer, neurons to process)

            Returns:
                new_tuple_states (list): New-generated state tuples
        """
        s, layer, neurons = tuple_state

        if (layer == self._num_layer - 1) and (len(neurons)==0):  # the last layer
            return [(s, layer, np.array([]))]

        new_tuple_states = []
        if neurons.shape[0] == 0: # neurons empty, go to the next layer
            if self.linearization or self.repair:
                over_app_set = self.reach_over_tuple(tuple_state)
                if self.verify_vzono(over_app_set):
                    return []

            W = self._W[layer+1]
            b = self._b[layer+1]
            s.affine_map(W, b)
            new_tuple_states.append((s, layer+1, np.arange(s.M.shape[0])))
        else: # not empty
            new_vfl_sets, new_neurons = relu.exact_reach(s, neurons)
            for vfl in new_vfl_sets:
                new_tuple_states.append((vfl, layer, new_neurons))

        return new_tuple_states






