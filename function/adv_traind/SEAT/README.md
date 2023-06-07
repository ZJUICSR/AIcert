# Self-Ensemble Adversarial Training(seat)
对CIFAR-10数据集进行分类。使用PyTorch框架，并实现了对抗训练.
具体来说，定义了一个包含ResNet18、PreActResNet18、SmallCNN和Wide_ResNet_Madry模型的字典，
根据输入参数选择其中一个模型进行训练。
定义一个AttackerPolymer类，用于生成对抗样本。
在训练过程中，使用SGD优化器和学习率调整策略，同时记录模型在测试集上的性能。
最后，保存最佳模型和教师模型的权重，并使用它们生成对抗样本并记录其性能。


## API
    seat(args_dict)

## 参数含义
```python
epochs：训练的轮数。
arch：选择使用的网络，可以选择 smallcnn、resnet18 或 WRN。
num_classes：分类的类别数。
lr：学习率。
loss_fn：损失函数。
epsilon：扰动的上限。
num-steps：最大扰动步数。
step-size：每步扰动的大小。
resume：是否继续训练。
out-dir：输出文件夹的路径。
ablation：消融研究。
```