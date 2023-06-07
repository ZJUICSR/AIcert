"""
These functions are used to load network models from .onnx files

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause


"""

import onnx
import torch
from torch import nn
import onnx2pytorch  # pip install onnx2pytorch


def load_ffnn_onnx(path):
    """
    Load ffnn model from a .onnx file

    Parameters:
        path (str): Path to the network model

    Returns:
        torch_model (pytorch): Network model
    """

    onnx_model = onnx.load(path)
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model)

    modules = list(pytorch_model.modules())[1:]
    new_modules = []
    for m in modules:
        if isinstance(m, torch.nn.ReLU) or isinstance(m, torch.nn.Linear):
            new_modules.append(m)

    torch_model = nn.Sequential(*new_modules)
    return torch_model


def save_onnx(torch_model, input_size, savepath):
    """
    Save network model to a .onnx model

    Parameters:
        torch_model (pytorch): Network model
        input_size (tensor): Size of the input image
        savepath (str): Path to save the model
    """

    x = torch.randn(10, input_size, requires_grad=True)
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      savepath,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})