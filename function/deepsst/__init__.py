from .DeepSst import *
# import .DeepSst as ds
import os.path as osp
CURR = osp.dirname(osp.abspath(__file__))

# ds.DeepSst(dataset='mnist',
#           pertube=0.05,
#           gpu=2,
#           filename=CURR+'/demo.json',
#           save_path=CURR,
#           modelname='LeNet',
#           path=CURR+'/mnist_lenet5.pth',
#           m_dir=CURR)

def run_deepsst(dataset, modelname, pertube, m_dir, out_path, logging=None):
    """
    dataset: 数据集名称
    modelname: 模型名称
    pertube: 敏感神经元扰动比例
    m_dir: 敏感度值文件位置
    """
    logging.info("Start running......")
    if m_dir=="":
        m_dir = CURR.rsplit('/',2)[0]+"/dataset/data/npy"
    path = CURR.rsplit('/',2)[0]+'/model/ckpt/mnist_lenet5.pth'
    res = DeepSst(dataset=dataset, modelname=modelname, pertube=pertube, m_dir=m_dir, path=path, save_path=out_path, logging=logging)
    logging.info("Finishing......")
    return res