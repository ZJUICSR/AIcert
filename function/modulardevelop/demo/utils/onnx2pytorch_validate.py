# import mxnet.contrib.onnx as onnx_mxnet
# import mxnet as mx
import numpy as np
import torch
import onnx
import onnx2pytorch as oi
from collections import namedtuple


# def construct_mxnext_model(onnx_file, test_input):
#     sym, arg, aux = onnx_mxnet.import_model(onnx_file)
#     data_names = [graph_input for graph_input in sym.list_inputs()
#                   if graph_input not in arg and graph_input not in aux]
#     print("Input Blob Names:", data_names)
#     mod = mx.mod.Module(symbol=sym, data_names=data_names, context=mx.cpu(), label_names=None)
#     print(sym)
#     # exit(0)
#     mod.bind(for_training=False, data_shapes=[(data_names[0], test_input.shape)], label_shapes=None)
#     mod.set_params(arg_params=arg, aux_params=aux, allow_missing=True, allow_extra=True)

#     Batch = namedtuple('Batch', ['data'])
#     # forward on the provided data batch
#     mod.forward(Batch([mx.nd.array(test_input)]))
#     output = mod.get_outputs()[0]
#     mo = output.asnumpy()
#     return mo


def construct_pytorch_model(onnx_file, test_input):
    onnx_model = onnx.load(onnx_file)
    if onnx_file == "densenet121.onnx":
        reconstruct_model = oi.DependencyModule(onnx_model, input_name="data_0")
    else:
        reconstruct_model = oi.DependencyModule(onnx_model)
    reconstruct_model.eval()
    i = torch.from_numpy(test_input).float()
    o = reconstruct_model(i).detach().numpy()
    return o


def test_onnx_model(onnx_file):
    print("=" * 80)
    print(onnx_file, ":")

    test_input = np.random.randn(1, 3, 32, 32) / 10
    o = construct_pytorch_model(onnx_file, test_input)
    # mo = construct_mxnext_model(onnx_file, test_input)
    abs_error = np.absolute(mo - o)

    print(abs_error.max(), abs_error.mean(), abs_error.min())
    print(mo[0][:5])
    print(o[0][:5])


def main():
    # ok_onnx_model_files = [
    #     "googlenet.onnx",  # OK special padding setting case not supported by PyTorch MaxPool. with Softmax()
    #     "resnet18v2.onnx",  # OK
    #     "resnet34v2.onnx",  # OK
    #     "squeezenet1.1.onnx",  # OK
    #     "mobilenetv2-1.0.onnx",  # OK
    #     "alex_net.onnx",  # OK but max error is not small enough. with Softmax()
    #     "densenet121.onnx",  # OK but input_name is 'data_0', not '0' in onnx.graph.input
    #     "vgg16.onnx",  # OK
    #     # "inception_v2.onnx",         # TODO wrong output, with Softmax()
    #     # "inception_v1.onnx",         # TODO Gemm weight shape in runtime
    #     # "shuffle_net.onnx",          # TODO wrong output, maybe by transpose or Softmax()
    # ]
    # for model_file in ok_onnx_model_files:
    #     test_onnx_model(model_file)
    test_onnx_model('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/model.onnx')


if __name__ == '__main__':
    main()



