##接口说明
集成防御模块

###接口名称
integrated_defense.integrate.py文件中的integrated_defense函数

### 接口参数
- model              # 被加固模型
- dataloader         # 选择的原始数据集
- attack_method      # 列表类型，字符串，攻击类型，fgsm、mifgsm、bim、cw等,参见attack.gen_adv.py中的get_attack函数中攻击类型
- eps                # 攻击方法的扰动值
- device             # 运行设备类型
- train_epoch        # 每次对抗训练的训练轮数，int
- in_channel         # 数据通道数 int,
- data_size          # 数据形状大小 int,

### 返回值：
{"ori_acc"：原始准确率， “攻击方法1”：{"attack_acc": 攻击准确率, "defend_rate": 防御成功率, "defend_attack_acc": 防御后攻击成功率, "defend_model": {'model': 训练模型, 'denoiser': 降噪模型}}, ”攻击方法2“: {...}, ...}

### 调用示例：
参见integrate.py文件中的main下面的测试函数