import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import onnxruntime
# import keras2onnx
import autokeras as ak
# import onnx
from tensorflow.keras.models import load_model,Model
import tensorflow.keras as keras
import numpy as np
import copy
import tensorflow as tf
import tensorflow.keras.optimizers as O
import tensorflow.keras.layers as L
import tensorflow.keras.activations as A
import tensorflow.keras.initializers as I

def replace_intermediate_layer_in_keras(model, layer_id, new_layer_list):
    layers = [l for l in model.layers]
    x = layers[0].output
    for i in range(1,len(layers)):
        if i == layer_id:
            for new_layer in new_layer_list:
                x = new_layer(x)
        else:
            x = layers[i](x)
    try:
        new_model = Model(input=layers[0].input, output=x)
    except:
        new_model = Model(inputs=layers[0].input, outputs=x)
    # new_model = Model(layers[0].input, x)
    return new_model

model1=load_model('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/best_model.h5',custom_objects=ak.CUSTOM_OBJECTS)
model=load_model('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/best_model-modify-0803.h5')
model1.save('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/0925/tfmodel_1')
model.save('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/0925/tfmodel_modify')
print(1)
# model.layers.pop(0)
# model.summary()
# newInput = keras.Input(shape=model1.input_shape)#keras.Input(shape=model1.layers[0].input_shape[0])    # let us say this new InputLayer
# newOutputs = model(newInput)
# newModel = Model(newInput, newOutputs)
# newModel.summary()

model=load_model('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/best_model.h5',custom_objects=ak.CUSTOM_OBJECTS)
pass_sign=True
tmp_weight_list=[]
for i in range(len(model.layers)):
    if pass_sign and not hasattr(model.layers[i],'layers'):
        continue
    if not pass_sign:
        new_config=copy.deepcopy(model.layers[i].get_config())
        outputs=model.layers[i].__class__(**new_config)(outputs)
        tmp_weight=model.layers[i].get_weights()
        if tmp_weight!=[]:
            tmp_weight_list.append((layer_length,tmp_weight))
        layer_length+=1
        continue
    pass_sign=False
    # inputs=model.layers[i].layers[0].input
    # outputs=model.layers[i].layers[0].output
    # for j in range(1,len(model.layers[i].layers)):
    #     outputs = model.layers[i].layers[j](outputs)
    # new_model = keras.Model(inputs=inputs, outputs=outputs, name="new_model")
    new_model=model.layers[i]
    new_model.layers.pop(0)
    inputs=new_model.inputs[0]
    outputs=new_model.outputs[0]
    layer_length=len(new_model.layers)
    print(1)
new_model_1 = Model(inputs, outputs)

for tmpw in tmp_weight_list:
    new_model_1.layers[tmpw[0]].set_weights(tmpw[1])
    
new_model_1.save('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/best_model-modify.h5')


# new_model = tf.keras.Sequential()
# pass_sign=True
# for i in range(len(model.layers)):
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
            