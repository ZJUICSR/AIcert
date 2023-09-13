import os, json 

def run_framework_test_exec(model, framework, out_path, logging=None):
    bl = ""
    for i in framework:
        if(i == 'PyTorch'):
            i =  'torch'
        elif (i == 'PaddlePaddle'):
            i= 'paddle'
        bl = bl + i.lower() + "-"
        
    print(bl[:-1])
    # 加载镜像
    logging.info("加载镜像环境中...")
    # os.system("docker load -i framework_test:1.1-args.tar")
    # 执行镜像，挂载目录
    logging.info("新建容器，配置环境中...")
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    os.system("sudo docker run --name framework_args -v " +out_path+":/root/framework_test/frame_test_result -it -d framework_test:1.1_args")
    # 执行命令，结果保存
    logging.info("执行开发环境安全结构度量中...")
    # os.system("sudo docker exec -it framework_args bash -c 'cd /root/framework_test; source /etc/profile ; conda activate test_env ; python __init__argparse.py --model "+model+" -bl " +bl[:-1]+" '>> " +out_path+"_log.txt")
    os.system("sudo docker exec -it framework_args bash -c 'cd /root/framework_test; source /etc/profile ; conda activate test_env ; python __init__argparse.py --model "+model+" --backend_list " +bl[:-1]+" ' ")
    logging.info("结果保存至"+out_path)
    os.system("sudo docker rm -f framework_args")
    # not_exist_flag = True
    # while (not_exist_flag):
    #     if os.path.exists(out_path+"/report.json"):
    #         with open(out_path+"/report.json") as f:
    #             res = json.load(f)
    #             not_exist_flag = False
    #             break    
    #     else:
    #         continue
    
    with open(out_path+"/report.json") as f:
        res = json.load(f)
      
    res["out_path"] = out_path
    logging.info("开发环境安全结构度量完成")
    return res