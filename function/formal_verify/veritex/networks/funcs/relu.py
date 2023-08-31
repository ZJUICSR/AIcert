"""
These functions are used for reachability analysis in the ReLU function

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause

TODO: Add more linearization methods
"""

import numpy as np


def exact_reach(s, neurons):
    """
    Process the input set 's' with the first relu neuron from 'neurons'

        Parameters:
            s (FVIM or Flattice): Input set to neurons
            neurons (np.ndarray): ReLU neurons to further process s

        Returns:
            sets (list): New-generated sets
            new_neurons (np.ndarray): Rest of neurons to process in the future
    """

    new_neurons = neurons
    # An empty 'neurons' indicates the reachability analysis of this relu layer is done
    if neurons.shape[0] == 0:
        return [s], new_neurons

    new_neurons, new_neurons_neg = get_valid_neurons(s, neurons)
    s.affine_map_negative(new_neurons_neg)

    if new_neurons.shape[0] == 0:
        return [s], new_neurons

    sets = split(s, new_neurons[0])
    new_neurons = new_neurons[1:]
    return sets, new_neurons


def split(s, l):
    """
    Split the input set into at most two subsets in the exact reachability analysis

        Parameters:
            s (FVIM or Flattice): An input set
            l (int): Index of the relu neuron in the layer

        Returns:
            outputs (list): Output reachable sets
    """

    outputs = []
    sub_pos, sub_neg = s.relu_split(l)
    if sub_pos:
        outputs.append(sub_pos)
    if sub_neg:
        outputs.append(sub_neg)

    return outputs


def layer_linearize(s):
    """
    Process 's' with respect to a relu neuron using relu linearization

        Parameters:
            s (Vzono): An input reachable set

        Returns:
            s (Vzono): An output reachable set
    """

    neurons_neg_pos, neurons_neg = s.get_valid_neurons_for_over_app()
    s.base_vertices[neurons_neg,:] = 0
    s.base_vectors[neurons_neg,:] = 0

    if neurons_neg_pos.shape[0] == 0:
        return s

    base_vertices = s.base_vertices[neurons_neg_pos,:]
    vals = np.sum(np.abs(s.base_vectors[neurons_neg_pos,:]), axis=1, keepdims=True)
    ubs = np.max(base_vertices,axis=1, keepdims=True) + vals
    lbs = np.min(base_vertices, axis=1, keepdims=True) - vals
    M = np.eye(s.base_vertices.shape[0])
    b = np.zeros((s.base_vertices.shape[0], 1))
    base_vectors_relax = np.zeros((s.base_vertices.shape[0],len(neurons_neg_pos)))

    A = ubs / (ubs - lbs)
    epsilons = -lbs*A/2
    M[neurons_neg_pos, neurons_neg_pos] = A[:,0]
    b[neurons_neg_pos] = epsilons
    base_vectors_relax[neurons_neg_pos, range(len(ubs))] = epsilons[:,0]

    new_base_vertices = np.dot(M,s.base_vertices) + b
    new_base_vectors = np.concatenate((np.dot(M, s.base_vectors), base_vectors_relax), axis=1)
    s.base_vertices = new_base_vertices
    s.base_vectors = new_base_vectors

    return s


def get_valid_neurons(s, neurons):
    """
    Identify the relu neurons whose both negative and positive input ranges are spanned by the input set
    and the neurons whose only negative input range is spanned by the input set

    Parameters:
        s (FVIM or Flattice): Input reachable set
        neurons (np.ndarray): A list of neuron indices

    Returns:
        valid_neurons_neg_pos (np.ndarray): ReLU neurons whose both negative and positive input ranges are spanned by the input set
        valid_neurons_neg (np.ndarray): Neurons whose only negative input range is spanned by the input set
    """

    assert neurons.shape[0]!=0

    elements = np.dot(s.vertices,s.M[neurons,:].T)+s.b[neurons,:].T
    flag_neg = (elements <= 0)
    temp_neg = np.all(flag_neg, 0)
    temp_pos = np.all(elements>=0, 0)
    temp_sum = temp_neg + temp_pos
    indx_neg_pos = np.asarray(np.nonzero(temp_sum == False)).T[:,0]
    valid_neurons_neg_pos = neurons[indx_neg_pos]
    indx_neg = np.asarray(np.nonzero(temp_neg)).T[:,0]
    valid_neurons_neg = neurons[indx_neg]

    return valid_neurons_neg_pos, valid_neurons_neg


