import os, sys, json
sys.path.append(os.path.dirname(__file__))
# utls_file = os.path.join(os.path.dirname(__file__),'utils')
# sys.path.append(utls_file)
from demo import run_search, run_convert

def run_modulardevelop(dataset,model,tuner,init, epoch,iternum, device, out_path, logging=None):
    logging.info("开始执行模块化开发功能......")
    os.system("bash -c 'source ~/anaconda3/etc/profile.d/conda.sh &&  conda activate test && python function/modulardevelop/demo/run_search.py --data "+dataset+" --block_type "+model+" --tuner "+tuner+" --init "+init+ " --epoch "+str(epoch)+" --iternum "+str(iternum)+" --save_dir "+out_path+"'")
    logging.info("模型搜索完成......")
    with open(out_path+"/search_result.json",'r') as f:
        res = json.load(f)
        # print(res)
    logging.info("将搜索模型保存为Tensorflow格式......")
    res["target_tensorflow"] = os.path.join(out_path,"best_model.h5")
    # 执行框架转换 PyTorch, PaddlePaddle
    logging.info("将搜索模型保存为PyTorch格式......")
    os.system("bash -c 'source ~/anaconda3/etc/profile.d/conda.sh &&  conda activate test && python function/modulardevelop/demo/run_convert.py --data "+dataset+" --model_path "+out_path+"/best_model.h5 --target_bk torch  --save_dir "+out_path+"/target_torch'")
    res["target_torch"] = os.path.join(out_path,"target_torch","best_model.pth")
    logging.info("将搜索模型保存为PaddlePaddle格式......")
    os.system("bash -c 'source ~/anaconda3/etc/profile.d/conda.sh &&  conda activate test && python function/modulardevelop/demo/run_convert.py --data "+dataset+" --model_path "+out_path+"/best_model.h5 --target_bk paddle  --save_dir "+out_path+"/target_paddle'")
    res["target_paddle"] = os.path.join(out_path,"target_paddle")
    # 退出到项目环境
    os.system("bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda deactivate")
    # print(res)
    return res