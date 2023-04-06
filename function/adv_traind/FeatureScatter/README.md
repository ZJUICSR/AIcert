# 鲁棒性增强、检测接口

## RobustEnhance()
鲁棒性增强接口，输出鲁棒增强后的模型和增强后的模型分类准确率
### 参数说明：
    --resume：是否从之前的检查点恢复训练，默认为 True。
    --lr：学习率，默认为 0.1。
    --adv_mode：对抗训练模式，默认为 'feature_scatter'。
    --model_dir：模型路径，默认为 '/data/user/WZT/models/feature_scatter_cifar10/resnet'。
    --init_model_pass：初始化模型的检查点，默认为 '-1'，表示从头开始训练；如果为 K，则从第 K 个检查点开始训练。
    --max_epoch：最大训练轮数，默认为 200。
    --save_epochs：每隔多少轮保存一次模型，默认为 100。
    --decay_epoch1：学习率第一次衰减的轮数，默认为 60。
    --decay_epoch2：学习率第二次衰减的轮数，默认为 90。
    --decay_rate：学习率衰减的速率，默认为 0.1。
    --batch_size_train：训练时的批量大小，默认为 128。
    --momentum：动量，默认为 0.9。
    --weight_decay：权重衰减，默认为 2e-4。
    --log_step：每隔多少轮记录一次日志，默认为 10。
    --num_classes：数据集的类别数，默认为 10。
    --image_size：图像的大小，默认为 32。
    --dataset：数据集的名称，默认为 'cifar10'。
## RobustTest()
鲁棒性检验接口，输入鲁棒增强后的模型，输出attack之后的分类准确率
### 参数说明：
    --resume: 是否从之前的检查点恢复训练。
    --attack: 是否使用对抗性训练。
    --model_dir: 模型路径。
    --init_model_pass: 初始模型检查点。
    --attack_method: 对抗性攻击方法（natural, pdg 或 cw）。
    --attack_method_list: 对抗性攻击方法列表，以逗号分隔。
    --log_step: 训练日志步长。
    以下是与数据集相关的参数：
    --num_classes: 类别数。
    --dataset: 数据集名称。
    --batch_size_test: 测试时的批量大小。
    --image_size: 图像大小。
## 接口调用:test.py