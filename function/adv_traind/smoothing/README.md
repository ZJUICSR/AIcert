

# via Randomized Smoothing Adversarial Robustness

## train2.py

使用数据增强技术，包括在输入数据中添加高斯噪声，从而提高模型的鲁棒性。
从数据集中加载数据，并使用SGD优化器和StepLR学习率调度器进行训练。
在训练过程中，使用数据增强技术，在输入数据中添加高斯噪声，提高模型的鲁棒性。
训练过程中，每个epoch会保存一个checkpoint，包括模型的状态、优化器的状态和当前epoch数。
同时，训练过程中会记录训练和测试的loss和accuracy，并写入一个日志文件中。
具体实现细节可以参考train.py文件中的代码。

### 训练参数含义

```
--dataset：指定数据集的名称，可选值为cifar10、cifar100和imagenet。
--arch：指定网络架构的名称，可选值为resnet20、resnet32、resnet44、resnet50、resnet110和wrn28_10。
--outdir：指定保存模型和训练日志的文件夹路径。
--workers：指定用于数据加载的进程数。
--epochs：指定训练的总epoch数。
--batch：指定batch size的大小。
--lr：指定初始学习率。
--lr_step_size：指定学习率下降的步数。
--gamma：指定学习率下降的倍数。
--momentum：指定SGD优化器的动量参数。
--weight-decay：指定权重衰减的系数。
--noise_sd：指定高斯噪声的标准差{0.0,0.25,0.5,1.0}。
--gpu：指定使用的GPU的编号。
--print-freq：指定训练过程中日志输出的频率。
```

具体使用train2.mian()接口参数。含义：将在 σ=0.50 的高斯数据增强下在cifar10 上训练 ResNet-50

### 接口使用参数

```python
args_dict = {
	'dataset': 'cifar10',#数据集
	'arch': 'resnet50',#模型原型
	'outdir': '/data/user/WZT/models/smoothing/cifar10/resnet110/noise_0',#训练模型输出路径
	'batch': 400, #训练批次大小
	'noise': 0.50 # 用于数据增强的高斯噪声的标准差0.0 0.25 0.5 1.0
}
```

## certify2.py

对数据集上的样本进行平滑分类器的评估。
代码加载一个已经训练好的基分类器，并使用该分类器创建一个平滑分类器。
代码迭代遍历数据集中的每个样本，对每个样本进行平滑分类器的预测，并将结果写入输出文件中。
代码使用Smooth类中的certify方法来计算预测的半径和置信度，并将结果写入输出文件中。

certify2.py证明*g*对大量输入的稳健性

### 接口使用参数

```python
args_dict = {
    'dataset': 'cifar10',  # 数据集
    'base_classifier': '/data/user/WZT/models/smoothing/cifar10/resnet110/noise_0.00/checkpoint.pth.tar',  # 模型文件
    'sigma': 0.50,  # 噪声水平
    'outfile': '/data/user/WZT/models/smoothing/certification_output/result',  # 输出g对一堆输入进行预测的结果文件
    'batch': 400,  # 训练批次大小
    'skip': 100,  # 每隔一百个图像
}
```

## predict2.py

在给定数据集上运行基础分类器并进行预测的脚本。

加载基础分类器并创建平滑分类器。
输出文件并迭代数据集中的每个样本。
只对每个 args.skip 个样本进行认证，并在 args.max 个样本后停止。
对于每个样本，将其输入到平滑分类器中进行预测，并记录预测结果、是否正确以及所用时间。
将这些信息写入输出文件中。

predict2.py含义：*g*对一堆输入进行预测



### 接口使用参数

```python
args_dict = {
    'dataset': 'cifar10',  # 数据集
    'base_classifier': '/data/user/WZT/models/smoothing/cifar10/resnet110/noise_0.00/checkpoint.pth.tar',  # 模型文件
    'sigma': 0.50,  # 噪声水平
    'outfile': '/data/user/WZT/models/smoothing/prediction_outupt/result',  # 输出g对一堆输入进行预测结果
    'batch': 400,  # 训练批次大小
    'skip': 100,  # 每隔一百个图像
    'alpha': 0.001,  #
    'N': 100000
}
```

code为接口代码文件 train2.py，certify2.py, predict2.py

data为数据集路径包含cifar10数据集

model为模型文件保存路径

certification_output为certify2.py输出路径

prediction_outupt为predict2.py输出路径







