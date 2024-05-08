from keras.preprocessing import image
from tensorflow.keras.models import load_model
import argparse, keras
import autokeras as ak
import os, json
import os.path as osp
import numpy as np
import tensorflow as tf

def process_img(image_path):
    # IF Cifar10
    img = image.load_img(image_path, target_size=(32, 32))
    x = image.img_to_array(img)
    # print(img)
    x = np.expand_dims(x, axis=0)
    
    
    x = x.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x[:, :, :, i] = (x[:, :, :, i] - mean[i]) / std[i]
        
    return x


def inference_online(image_path, model_path):

    # label map for Cifar10
    label_map = {
        "0": "airplane",
        "1": "automobile",
        "2": "bird",
        "3": "cat",
        "4": "deer",
        "5": "dog",
        "6": "frog",
        "7": "horse",
        "8": "ship",
        "9": "truck"
    }
    #if mnist
    # label_map =  {
    #     "0": "0",
    #     "1": "1",
    #     "2": "2",
    #     "3": "3",
    #     "4": "4",
    #     "5": "5",
    #     "6": "6",
    #     "7": "7",
    #     "8": "8",
    #     "9": "9"
    # }


    model = load_model(model_path,custom_objects=ak.CUSTOM_OBJECTS)
    # print(model.summary())

    img = process_img(image_path)
    pre_y = model.predict(img)
    # print(pre_y)
    pred_label = label_map[str(np.argmax(pre_y[0]))]
    print(pred_label)
    return pred_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Framework Test')
    parser.add_argument('--image_path', default='/data/home/Wenjie/Project/AIcert/dataset/data/ckpt/upload.jpg', help='image_path')
    parser.add_argument('--model_path', default='/data/home/Wenjie/Project/AIcert/model/ckpt/best_model.h5',  help='search result')
    args = parser.parse_args()
    res = inference_online(args.image_path,args.model_path)
    
    if osp.exists("./output/inference.json"):
        os.remove("./output/inference.json")
    with open("./output/inference.json",'w') as f:
        json.dump({'imageLabel': res}, f)
        
    
    