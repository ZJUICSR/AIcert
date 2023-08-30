"""
These functions are used to visualize the 2D or 3D reachable domain of a DNN with respect to a safety property.

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause

TODO: Add more visualization functions
"""

import sys
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import argparse

from veritex.utils.load_nnet import NNet
from veritex.networks.ffnn import FFNN
from veritex.utils.plot_poly import plot_polytope2d
from veritex.methods.worker import Worker
from veritex.methods.shared import SharedState
from veritex.utils.load_onnx import load_ffnn_onnx
from veritex.utils.sfproperty import Property
from veritex.utils.vnnlib import vnnlib_to_properties
import torch


def run(prop_list, network_path, dims, savename, figsize=(2.0, 2.67)):
    """
    Plot reachable domains of the network model on the safety properties

    Parameters:
        prop_list (list): A list of paths to safety properties
        network_path (str): Path to the network model
        dims (list): Dimensions to project reachable domains on
        savename (str): Path to save figures
    """
    if network_path[-4:] == 'onnx':
        torch_model = load_ffnn_onnx(network_path)
        input_num, output_num = torch_model[0].in_features, torch_model[-1].out_features
    elif network_path[-2:] == 'pt':
        torch_model = torch.load(network_path)
        input_num, output_num = torch_model[0].in_features, torch_model[-1].out_features
    elif network_path[-4:] == 'nnet':
        model = NNet(network_path)
        biases = [np.array([bia]).T for bia in model.biases]
        func = ['ReLU'] * (len(biases) - 1)  # default activation function for .nnet
        torch_model = [model.weights, biases, func]
        input_num, output_num = model.num_inputs(), model.num_outputs()
    else:
        sys.exit('Network file is not found!')

    # Output dimensions to project on
    dim0, dim1 = dims

    # Extract safety properties
    if isinstance(prop_list[0],Property):
        properties = prop_list
    else:
        properties = []
        for prop in prop_list:
            temp = vnnlib_to_properties(prop, input_num, output_num)
            properties.extend(temp)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    dnn0 = FFNN(torch_model, unsafe_inputd=True, exact_outputd=True)

    # Compute the exact unsafe output domains of the network model over safety properties
    for prop in properties:
        dnn0.set_property(prop)

        processes = []
        results = []
        num_processors = multiprocessing.cpu_count()
        shared_state = SharedState(prop, num_processors)
        one_worker = Worker(dnn0)
        for index in range(num_processors):
            p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        while not shared_state.outputs.empty():
            results.append(shared_state.outputs.get())

        output_sets = [item[1] for item in results]
        unsafe_sets = []
        for item in results:
            if item[0]: unsafe_sets.extend(item[0])

        for _ in range(len(output_sets)):
            item = output_sets.pop()
            # Compute the vertices of all output reachable sets
            out_vertices = np.dot(item.vertices, item.M.T) + item.b.T
            plot_polytope2d(out_vertices[:, [dim0, dim1]], ax, color='b', alpha=1.0, edgecolor='k', linewidth=0.0)

        for _ in range(len(unsafe_sets)):
            item = unsafe_sets.pop()
            # Compute the vertices of unsafe output reachable sets
            out_unsafe_vertices = np.dot(item.vertices, item.M.T) + item.b.T
            plot_polytope2d(out_unsafe_vertices[:, [dim0, dim1]], ax, color='r', alpha=1.0, edgecolor='k',
                            linewidth=0.0)

    ax.autoscale()
    ax.set_xlabel('$y_' + str(dim0+1) + '$', fontsize=16)
    ax.set_ylabel('$y_' + str(dim1+1) + '$', fontsize=16)
    # plt.title('Exact output reachable domain (blue) & Unsafe domain (red) on'+' Property '+args.property, fontsize=18, pad=20)

    plt.savefig(savename + '.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting of reachable domains')
    parser.add_argument('--property', nargs='+', required=False)
    parser.add_argument('--network_path', type=str, required=False)
    parser.add_argument('--savename', type=str, required=False)
    parser.add_argument('--dims', nargs='+', type=int, default=(0, 1))
    args = parser.parse_args()
    print(args.property)
    run(args.property, args.network_path, args.dims, args.savename)



