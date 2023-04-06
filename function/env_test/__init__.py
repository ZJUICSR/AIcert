import os
from .env_test import ENVT
from .gpu_test import framework_test
import sys
import os.path as osp
import json

ROOT = osp.dirname(osp.abspath(__file__))
sys.path.append(ROOT)


def run_env(
        json_path,
        method,
        save_dir,
        cve_path
    ):
    '''
    Input
    json_path: 保存结果json文件的路径，字符串格式，json中每个key都被一个功能块使用，每个key中保存一个字典，保存功能
    块的结果
    method：从输入页面获得输入，字符串格式，可输入"hard" "all" "soft"

    Output
    返回包括该模块结果的json文件，该文件也保存在了json_path路径
    '''
    env = ENVT(json_path=json_path,save_dir=save_dir,cve_path=cve_path
              )
    # return a json file
    return env.detection(method=method)

def run_framework(
        save_dir,
        json_path,
        frame,
        version
    ):
    '''
    Input
    save_dir：保存结果pkl文件的目录
    json_path: 保存结果json文件的路径，字符串格式，json中每个key都被一个功能块使用，每个key中保存一个字典，保存功能
    块的结果
    frame：需要测试是否适配的框架名称，可输入"tensorflow" "pytorch" "paddlepaddle"
    version：需要测试是否适配的框架版本名称，字符串格式。例如"1.7.1"

    Output
    返回包括该模块结果的json文件，该文件也保存在了json_path路径
    '''
    message,cuda_version = framework_test(frame,version)
    if os.path.exists(json_path):
        with open(json_path,'r') as f:
            json_result = json.load(f)
        detection_result_path=os.path.join(save_dir,json_result['env_test']['detection_result'])
    else:
        print('Error! no json path!!!')
        json_result={}
        detection_result_path=os.path.join(save_dir,'keti2/env_test_result/detection.json')
    with open(detection_result_path,'r') as f:
        detection_result = json.load(f)
    frame_test = {"Framework Adaptation Result":message,"Current Cuda Version":cuda_version}
    detection_result.update(frame_test)
    
    with open(detection_result_path,"w") as f:
        f.write(json.dumps(detection_result,ensure_ascii=False,indent=4,separators=(',',':')))
    # return a json file
    return json_result


def run_env_frame(method, frame, version):
    out_path = ROOT+"/env"
    cve_path = ROOT[:-17]+"/model/valid_extract.pkl"
    json_path = os.path.join(out_path, "env_results.json")
    save_dir= os.path.join(out_path, "env_test_result")
    print("Analyzing Evironment.....")
    run_env(json_path=json_path, method=method,cve_path=cve_path,save_dir=save_dir)
    print("Finish Evironment!! Analyzing Framework.....")
    res = run_framework(save_dir=out_path,json_path=json_path, frame=frame,version=version)
    print(f"Finish Framework!! Please Check results in {json_path}")
    return res

def run(params):
    json_path = os.path.join(params["out_path"], "keti2/keti2.json")
    save_dir= os.path.join(params["out_path"], "keti2/env_test_result")
    print("Analyzing Evironment.....")
    run_env(json_path=json_path, method='hard',cve_path=params["cve_path"],save_dir=save_dir)
    print("Finish Evironment!! Analyzing Framework.....")
    run_framework(save_dir=params["out_path"],json_path=json_path, frame=params["frame"],version=params["version"])
    print(f"Finish Framework!! Please Check results in {json_path}")

if __name__=='__main__':
    params={}
    params["out_path"]="./env"
    # 输出结果所在文件夹
    params["cve_path"]="./valid_extract.pkl"
    params["frame"] = "pytorch"#frame：需要测试是否适配的框架名称，可输入"tensorflow" "pytorch" "paddlepaddle"
    params["version"] = '1.7.1'#version：需要测试是否适配的框架版本名称，字符串格式。例如"1.7.1"
    # "2.10.0"# fail
    run(params)

    '''
    import pickle
    with open('.\env\keti2\env_test_result\system_message.pkl', 'rb') as f:
        sys_env = pickle.load(f)
    '''