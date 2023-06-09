from .deeplogic import *
import os.path as osp
CURR = osp.dirname(osp.abspath(__file__))


def run_deeplogic(dataset, modelname, out_path, logging=None):
    """
    dataset: 数据集名称
    modelname: 模型名称
    out_path: 输出路径
    logging: 日志
    """
    if not osp.exists(out_path):
        os.mkdir(out_path)
    logging.info("Start running.......")
    # default_out_path=CURR.rsplit('/',2)[0]+'/test_case_select/result'
    res=evaluate_deeplogic(dataset, modelname, out_path=out_path, logging=None)
    logging.info("Finishing.......")
    return res

if __name__=='__main__':
    res=evaluate_deeplogic('cifar10', 'vgg16', out_path='test_case_select/result', logging=None)
    print(res)