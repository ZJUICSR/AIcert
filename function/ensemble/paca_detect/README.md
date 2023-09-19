##接口说明

###接口名称
paca_detect.paca.py文件中的paca_detect函数


### 接口参数
- ori_model # pytorch模型
- ori_dataloader     # 选择的原始数据集
- attack_info        # 列表类型，[{"method": attack1, "dataloader": dataloader1},
                                 {"method": attack2, "dataloader": dataloader2},
                                   ...]， 其中，method的值为攻击类型（FGSM、DeepFool等），
                                   dalaloader的值为原始数据集在该攻击下产生的dataloader
- param_hash         # 参数hash值，类型：Str，用于保存和加载测试结果缓存
- device             # 设备类型：默认值:'cuda',
- save_path          # 结果存储位置
- log_func           # 日志函数，如果需要在执行过程中输出记录过程日志，传入记录日志函数,日志函数的参数类型为：Str

### 安装依赖：
- torch

### 调用示例：
参见paca_detect.paca.py文件中的demo函数