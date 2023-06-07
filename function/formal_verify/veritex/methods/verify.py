"""
These functions are used to verify the safety of a DNN over safety properties

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause

"""

from veritex.networks.ffnn import FFNN
import multiprocessing as mp
from veritex.methods.worker import Worker
from veritex.methods.shared import SharedState
from veritex.utils.load_onnx import load_ffnn_onnx
from veritex.utils.vnnlib import vnnlib_to_properties
from veritex.utils.sfproperty import Property
from veritex.utils.load_nnet import NNet
import multiprocessing
import numpy as np
import logging
import time
import torch
import sys
import argparse


def run(properties_list, network_path, netname, propnames, linearization=True):
    """
    Run safety verification of the network over a set of safety properties.

    Parameters:
        properties_list (list): Safety properties to verify
        network_path (str): Path to load the network model
        netname (str): Name to specify the network model
        propnames (list): Names to specify the safety properties
        linearization (bool): Linearization of activation functions.

    """

    # Creating and Configuring Logger
    logger = logging.getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    Log_Format = logging.Formatter('%(levelname)s %(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    propstr = ''
    for na in propnames: propstr += '_'+na
    file_handler = logging.FileHandler(f'verify_network_{netname}_on_property{propstr}.log', 'w+')
    file_handler.setFormatter(Log_Format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(Log_Format)
    logger.addHandler(console_handler)

    num_processors = multiprocessing.cpu_count()

    # load network
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

    # extract the safety properties
    if isinstance(properties_list[0], Property):
        properties = properties_list
    else: # extract safety properties from vnnlib files
        properties = []
        for prop in properties_list:
            temp = vnnlib_to_properties(prop, input_num, output_num, set_type='FVIM')
            properties.extend(temp)

    # configure the verification
    dnn0 = FFNN(torch_model, verification=True, linearization=linearization)

    # run safety verification
    for n, prop in enumerate(properties):
        t0 = time.time()
        unsafe = False
        dnn0.set_property(prop)

        processes = []
        shared_state = SharedState(prop, num_processors)
        one_worker = Worker(dnn0)
        for index in range(num_processors):
            p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        while not shared_state.outputs.empty():
            unsafe = shared_state.outputs.get()

        logging.info('')
        logging.info(f'Network {netname} on the property {propnames[n]}')
        logging.info(f'Unsafe: {unsafe}')
        logging.info(f'Running Time: {time.time() - t0} sec')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting of reachable domains')
    parser.add_argument('--property', nargs='+', required=True)
    parser.add_argument('--property_name', nargs='+', required=True)
    parser.add_argument('--linearize', action='store_false')
    parser.add_argument('--network_path', type=str, required=True)
    parser.add_argument('--network_name', type=str, required=True)
    args = parser.parse_args()
    run(args.property, args.network_path, args.network_name, args.property_name, linearization=args.linearize)





