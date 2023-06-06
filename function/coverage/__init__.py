# from coverage_layer import * # 以此为执行脚本时使用
from .coverage_layer import *
# from coverage_neural import *
from .coverage_neural import *
import os.path as osp
CURR = osp.dirname(osp.abspath(__file__))

def run_coverage_neural_func(dataset, model, k, N, out_path, logging):
    """
    dataset: 数据集名称
    model: 模型名称
    k: 激活阈值
    
    model_path: 模型路径
    """
    logging.info("Starting running......")
    model_path = CURR.rsplit('/',2)[0]+"/model/ckpt/lenet5_dict.pth"
    # res = run_visualize_neural(dataset=dataset, model_type=model, k=0.1, model_path=model_path, outputdir=out_path, logging=logging,number_of_image=N)
    res = run_visualize_neural(dataset=dataset, model_type=model, k=k, model_path=model_path, outputdir=out_path, logging=logging,number_of_image=N)
    logging.info("Finishing......")
    return res

def run_coverage_layer_func(dataset, model, k, N, out_path, logging):
    """
    dataset: 数据集名称
    model: 模型名称
    k: 激活阈值
    model_path: 模型路径
    """
    logging.info("Starting running......")
    model_path = CURR.rsplit('/',2)[0]+"/model/ckpt/lenet5_dict.pth"
    res = run_visualize_layer(dataset=dataset, model_type=model, k=k, model_path=model_path, outputdir=out_path, logging=logging,number_of_image=N)
    logging.info("Finishing......")
    return res

if __name__ == "__main__":
    # res = run_coverage_layer_func
    model_path = CURR.rsplit('/',2)[0]+"/model/ckpt/vgg11_dict.pth"
    res = run_visualize_neural(dataset="cifar10", model_type="vgg11", k=0.1, model_path=model_path, number_of_image=100)
    # print(res)
    # res = run_visualize_layer(dataset="mnist", model_type="lenet5", k=0.1, model_path=model_path, number_of_image=100)
    print(res)