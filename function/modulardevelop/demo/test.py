import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import onnxruntime
# import keras2onnx
from tensorflow.keras.models import load_model,Model
import autokeras as ak
# import onnx
import numpy as np
import copy
import pickle


with open('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result2/0-0.97-861a10a6ee20/history.pkl', 'rb') as f:
    log_dict = pickle.load(f)

# with open('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/param_mnist_resnet.pkl', 'rb') as f:
#     log_dict_1 = pickle.load(f)
# print(1)
with open('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result3/best_history.pkl', 'rb') as f:
    log_dict_1 = pickle.load(f)
print(2)
# def replace_intermediate_layer_in_keras(model, layer_id, new_layer_list):
#     layers = [l for l in model.layers]
#     x = layers[0].output
#     for i in range(1,len(layers)):
#         if i == layer_id:
#             for new_layer in new_layer_list:
#                 x = new_layer(x)
#         else:
#             x = layers[i](x)
#     try:
#         new_model = Model(input=layers[0].input, output=x)
#     except:
#         new_model = Model(inputs=layers[0].input, outputs=x)
#     # new_model = Model(layers[0].input, x)
#     return new_model

# autoKeras_model=load_model('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result1/resnet20.h5')
# print(type(autoKeras_model))
# # output=autoKeras_model.predict(np.random.random((1,28,28,1)))
# print(1)

# tf 2.3 sequential model test

# # Note that when using the delayed-build pattern (no input shape specified),
# # the model gets built the first time you call `fit`, `eval`, or `predict`,
# # or the first time you call the model on some input data.
# import tensorflow as tf
# import tensorflow.keras.optimizers as O
# import tensorflow.keras.layers as L
# import tensorflow.keras.activations as A
# import tensorflow.keras.initializers as I
# # model = tf.keras.Sequential()
# # model.add(tf.keras.layers.Dense(8))
# # model.add(tf.keras.layers.Dense(1))
# # model.compile(optimizer='sgd', loss='mse')
# # model.build((None, 16))
# model=load_model('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/best_model.h5',custom_objects=ak.CUSTOM_OBJECTS)
# new_model = tf.keras.Sequential()
# pass_sign=True
# for i in range(len(model.layers)):
#     # elif hasattr(model.layers[i],'layers'):
#     #     tmp_list=[]
#     #     for j in  range(len(model.layers[i].layers)):
#     #         new_config=copy.deepcopy(model.layers[i].layers[j].get_config())
#     #         new_layer=model.layers[i].layers[j].__class__(**new_config)
#     #         tmp_list.append(new_layer)
#     #     new_model=replace_intermediate_layer_in_keras(model,i,tmp_list)
#     if pass_sign and not hasattr(model.layers[i],'layers'):
#         continue
#     new_config=copy.deepcopy(model.layers[i].get_config())
#     if not pass_sign:
#         new_model.add(model.layers[i].__class__(**new_config))
#         continue
#     pass_sign=False
#     input_sign=False
#     for lc in range(len(new_config['layers'])):
#         if 'input' in new_config['layers'][lc]['name']:
#             input_shape=new_config['layers'][0]['config']['batch_input_shape']
#             input_sign=True
#             continue
#         tmp_config=new_config['layers'][lc]['config']
#         if input_sign:
#             tmp_config['input_shape']=input_shape[1:]
#         new_model.add(model.layers[i].layers[lc].__class__(**tmp_config))
# new_model.build(input_shape)
# new_model.summary()
            

# model.save('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result1/seq.h5')
# seq_model=load_model('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result1/seq.h5')
# print(type(model))
# print(1)

# autoKeras_model1=load_model('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result1/best_model.h5',custom_objects=ak.CUSTOM_OBJECTS)

import onnx
from onnx2pytorch import ConvertModel
import torch

onnx_model = onnx.load('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/1011/best_model-modify.onnx')
pytorch_model = ConvertModel(onnx_model)
torch.save(pytorch_model, '/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/1011/best_model_onnx2pytorch.pth')

import torch
from onnx2torch import convert

# Path to ONNX model
onnx_model_path = '/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/1011/best_model-modify.onnx'
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)
torch.save(torch_model_1, '/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/1011/best_model_onnx2torch.pth')


from onnx_pytorch import code_gen
code_gen.gen('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/1011/best_model-modify.onnx', '/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/1011/best_model_onnx_pytorch')

# from x2paddle.convert import pytorch2paddle
# pytorch2paddle(module=torch_module,
#                save_dir="./pd_model",
#                jit_type="trace",
#                input_examples=[torch_input])

# import os
# from onnx_pytorch import code_gen
# os.makedirs("/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/tmp_torch_model")
# code_gen.gen("/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/model.onnx", "/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/tmp_torch_model")


# import numpy as np
# import onnx
# import onnxruntime
# import torch
# torch.set_printoptions(8)
# import sys
# sys.path.append('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/tmp_torch_model/')
# from model import Model

# model = Model()
# model.eval()
# inp = np.random.randn(1, 28, 28).astype(np.float32)
# with torch.no_grad():
#   torch_outputs = model(torch.from_numpy(inp))

# onnx_model = onnx.load("/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/model.onnx")
# sess_options = onnxruntime.SessionOptions()
# session = onnxruntime.InferenceSession(onnx_model.SerializeToString(),
#                                        sess_options)
# inputs = {session.get_inputs()[0].name: inp}
# ort_outputs = session.run(None, inputs)

# print(
#     "Comparison result:",
#     np.allclose(torch_outputs.detach().numpy(),
#                 ort_outputs[0],
#                 atol=1e-5,
#                 rtol=1e-5))
