##接口说明
集成防御模块

###接口名称
ensemble_defense.ens_defense.py文件中的ens_defense函数

(model, dataloader, methods=[], eps=1, device='cuda'):
### 接口参数
- model              # 被加固模型
- dataloader         # 选择的原始数据集
- method             # 列表类型，字符串，攻击类型，fgsm、mi-fgsm、bim、cw等,参见attack.gen_adv.py中的get_attack函数中攻击类型
- eps                # 攻击方法的扰动值
- device             # 运行设备类型
- train_epoch        # 每次对抗训练的训练轮数，int
- at_epoch           # 对抗训练次数 int,

### 返回值：
{"攻击方法1"：准确率， “攻击方法2”：准确率， ...}

### 调用示例：
参见ensemble_defense.ens_defense.py文件中的main下面的测试函数