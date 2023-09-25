import os
import json
import sys
sys.path.append(os.path.dirname(__file__))

def run(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu']
    from develop import model_generate,summarize_result
    if params['search']:
        model_generate(
            block_type=params['block_type'],
            search=params['search'],
            data=params['data'],
            save_dir=params['save_dir'],#结果保存到临时中转文件夹
            epoch=params['epoch'],
            tuner=params['tuner'],
            trial=params['trial'],
            gpu=params['gpu'],
            init=params['init'],
            iter_num=params['iter_num']
            )
        
        summarize_result(json_path=params['json_path'],save_dir=params['save_dir'])#临时中转文件的结果在这里处理。
    else:
        model_generate(
            block_type=params['block_type'],
            search=params['search'],
            data=params['data'],
            save_dir=params['save_dir'],
            epoch=params['epoch'],
            param_path=params['param_path'])
        

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Framework Test')
    parser.add_argument('--data', default='mnist',choices=['mnist','cifar10'], help='dataset')
    parser.add_argument('--block_type', default='vgg',choices=['mobilenet','vgg','resnet', 'vanilla'], help='model type')
    parser.add_argument('--init', default=None,choices=['normal','large','random'], help='init type for deepalchemy')
    parser.add_argument('--tuner', default='dream',choices=['dream','deepalchemy'], help='model search type')
    parser.add_argument('--epoch', default='2',  help='search epoch')
    parser.add_argument('--iternum', default='4',  help='trail times per epoch')
    parser.add_argument('--save_dir', default='./result', help='save path')
    # parser.add_argument('--json_path', default='./result/search_result.json', help='result json path')
    args = parser.parse_args()
    # 功能1：搜索模型
    test_param_1={
        'block_type':args.block_type,
        'search':True,
        'data':args.data,
        'save_dir':args.save_dir,# 包含所有搜索历史文件
        'json_path':args.save_dir+"/search_result.json",#搜索历史json文件保存的目录，可以在上一个save_dir中
        'epoch': int(args.epoch),
        'tuner': args.tuner,
        'trial': int(args.iternum),
        'gpu':'1',
        'init': args.init,
        'iter_num': int(args.iternum),
    }    
    # print(test_param_1)
    run(test_param_1)
    
    # # 功能2：从参数生成模型
    # test_param_2={
    #     'block_type':'resnet',
    #     'search':False,
    #     'data':'cifar',
    #     'save_dir':'./result3',
    #     'epoch':2,
    #     'param_path':'./param_mnist_resnet.pkl',#./param_mnist_resnet.pkl   ./param_cifar_xception.pkl
    #     'gpu':'1',
    # }
    # run(test_param_2)