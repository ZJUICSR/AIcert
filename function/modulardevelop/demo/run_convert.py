import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import sys
import shutil
sys.path.append(os.path.dirname(__file__))

def run_convert(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu']
    if os.path.exists(params['save_dir']):
        shutil.rmtree(params['save_dir'])
    os.makedirs(params['save_dir'])
    if params['target_bk']=='torch':
        from develop import torch_convert
        torch_model_path=torch_convert(params['model_path'],params['save_dir'],params['dataset'])
        print("Torch model is saved in {}.".format(os.path.abspath(torch_model_path)))
    elif params['target_bk']=='paddle':
        from develop import paddle_convert
        paddle_model_path=paddle_convert(params['model_path'],params['save_dir'])
        print("Paddle model is saved in {}.".format(os.path.abspath(paddle_model_path)))

if __name__=='__main__':
    # 需要先运行init.py得到一个模型，github无法上传较大的模型文件
    import argparse
    parser = argparse.ArgumentParser(description='Framework Test')
    parser.add_argument('--data', default='mnist',choices=['mnist','cifar10'], help='dataset')
    parser.add_argument('--model_path', default='./result/best_model.h5',  help='search result')
    parser.add_argument('--target_bk', default='torch', choices=['torch', 'paddle'],  help='convert to target backend')
    parser.add_argument('--save_dir', default='./result_c_v-4',   help='convert result')
    args = parser.parse_args()
    test_param_3={
        'model_path':args.model_path,
        'target_bk':args.target_bk, #可选['torch', 'paddle']
        'save_dir':args.save_dir, # ==========该文件夹需要是空文件夹！！每次跑会删除里面的东西！！==============
        'dataset':args.data, #可选['mnist','cifar10']
        'gpu':'1',
    }
    # test_param_3={
    #     'model_path':'./result/best_model.h5',
    #     'target_bk':'torch', #['torch', 'paddle']
    #     'save_dir':'./result_cifar_xception',
    #     'dataset':'mnist', #['mnist','cifar10']
    #     'gpu':'1',
    # }
    '''
     接口说明：
     model_path：输入模型路径，h5格式，可以先通过__init__.py的run(test_param_2)构建测试模型。
     target_bk：目标后端，目前可以将h5模型转换为torch和paddle两种格式
     save_dir：结果保存目录
     gpu：使用gpu的id
    '''
    if not os.path.exists(os.path.abspath(test_param_3['save_dir'])):
        os.makedirs(test_param_3['save_dir'])
    run_convert(test_param_3)