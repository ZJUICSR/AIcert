##接口说明
扰动过滤功能模块

###接口名称
CAFD.train_or_test_denoiser.cafd函数

target_model=model, dataloader=dataloader, method='fgsm', eps=1, channel=1, data_size=28, weight_adv=5e-3,
         weight_act=1e3, weight_weight=1e-3, lr=0.001, itr=10,
         batch_size=128, weight_decay=2e-4, print_freq=10, save_freq=2, device='cuda'
### 接口参数
- target_model       # 被攻击模型 nn.Module
- dataloader         # 选择的原始数据集 Dataloader
- method             # 攻击类型，字符串，攻击类型，fgsm、mi-fgsm、bim、cw等，参见attack.gen_adv.py中的get_attack函数中攻击类型
- in_channel         # 数据通道，mnist：1，cifar：3
- data_size          # 数据宽度，mnist：28，cifar：32
- eps                # 攻击方法的扰动值 float
- itr                # 训练轮数 int
- device             # 运行设备类型 str
- batch_size         # 训练数据集切分大小 int
- 其余参数可选择默认值

### 返回值：
- 扰动过滤后模型的预测准确率，类型：float

### 调用示例：
参见CAFD.train_or_test_denoiser.py文件中的main下面的测试函数