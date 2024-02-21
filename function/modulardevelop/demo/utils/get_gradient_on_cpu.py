import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from autokeras.engine import compute_gradient as cg
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import tensorflow
import autokeras as ak
print(tensorflow.__version__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute model gradient by cpu')
    parser.add_argument('--model','-m',default='./tmp/model.h5', help='model path')# 'auto' 'cust'
    parser.add_argument('--data_x','-dx',default='./tmp/x.npy', help='input')# 'auto' 'cust'
    parser.add_argument('--data_y','-dy',default='./tmp/y.npy', help='output')# 'auto' 'cust'
    parser.add_argument('--epoch','-ep',default=0, help='current training epoch')# 'auto' 'cust'
    parser.add_argument('--save_path','-sp',default='./tmp/gradient_weight.pkl', help='the path to save gradients and weights')# 'auto' 'cust'


    args = parser.parse_args()
   

    epoch=args.epoch
    model=load_model(args.model)
    trainingExample=np.load(args.data_x)
    trainingY=np.load(args.data_y)


    gradient_list=[]
    model_weights=model.get_weights()
    evaluated_gradients = cg.get_gradients(model, trainingExample, trainingY)
    #     except Exception as e:
    #         print(e)
    #         layer_name = model.layers[3].name
    #         # find embedding layer
    #         evaluated_gradients = cg.rnn_get_gradients(model,layer_name, trainingExample, trainingY)
        # x, y, sample_weight = model._standardize_user_data(trainingExample, trainingY)
        # #output_grad = f(x + y + sample_weight)
        # evaluated_gradients = f([x , y , sample_weight,0])
    for i in range(len(evaluated_gradients)):
        if isinstance(evaluated_gradients[i],np.ndarray):
            gradient_list.append(evaluated_gradients[i])
    print(os.path.exists(args.save_path))
    if not os.path.exists(args.save_path):
        os._exit(0)
    with open(args.save_path, 'rb') as f:  
        save_dict = pickle.load(f)
    # save_dict={}
    # save_dict['gradient']={}
    # save_dict['weight']={}
    
    save_dict['gradient'][epoch]=gradient_list
    save_dict['weight'][epoch]=model_weights
    with open(args.save_path, 'wb') as f:
        pickle.dump(save_dict, f)
