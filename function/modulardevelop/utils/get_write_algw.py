import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pickle
import sys
sys.path.append('./utils')
from load_test_utils import judge_dirs

# print(tensorflow.__version__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute model gradient by cpu')
    parser.add_argument('--dir','-d',default='/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/0-0.96-7baaa2a82b6c', help='model path')# 'auto' 'cust'

    args = parser.parse_args()
    
    
    arch,loss,grad,wgt=judge_dirs(args.dir)
    algw="{}-{}-{}-{}".format(arch,loss,grad,wgt)
    algw_path=os.path.join(args.dir,'algw.pkl')
    with open(algw_path, 'wb') as f:
        pickle.dump(algw, f)