# 基于[audo_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)的模型形式化验证接口

## 1. 环境安装

### 1.1 audo_LiRPA安装

1. 进入audo_LiRPA目录执行命令：
```
   cd auto_LiRPA\auto_LiRPA  
    python setup.py install  
```
3. python依赖库安装：
```
   pip install -r requirements.txt
```


### 1.2 第三方包约定

*输出文件的地址，在配置文件中写出，一般为绝对目录*

需包含文件以及说明如下：
1. requirements.txt - python依赖包文件


### 1.2 接口信息
1. params.json - 输入参数

```json
{'input_param': 
    {
        'model': mn_model,
        'dataset': ver_data,
        'n_class': 10,
        'up_eps': 0.1,
        'down_eps': 0.01,
        'steps': 5,
        'device': 'cpu',
        'output_path': 'output'
    }
}
```

2. params.json - 输出参数

```json
{
'interface': 'Verification',
'node': '中间结果可视化', 
'output_param': 
    {}
}
```

---
## 2. Element Definition

### 2.1 训练数据类型：
- image(mnist, cifar10， gtsrb, MTFL)
- language(SST-2)

### 2.2 模型类型：
- CNN、ResNet、DenseNet;
- LSTM、Transformer;
下载LSTM和Transformer预训练模型：  
cd model/  
mkdir auto_LiRPA_model  
cd auto_LiRPA_model/  
wget http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/trained/ckpt_lstm  
wget http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/trained/ckpt_transformer  



## 3. 接口位置
### 3.1 vision
- 接口文件位置：auto_LiRPA_verifiy/vision/verify.py
- 接口函数：verify()
- 调用示例：auto_LiRPA_verifiy/vision/test.py
- 模型数据接口：支持mnist，cifar，gtsrb数据集   
from auto_LiRPA_verifiy.vision.data import get_mnist_data(获取mnist数据), get_cifar_data获取cifar数据), get_gtsrb_data(获取gtsrb交通数据), get_MTFL_data(获取人脸识别数据集)  
from auto_LiRPA_verifiy.vision.model import get_mnist_cnn_model（针对mnist的CNN模型）, get_cifar_resnet18（针对cifar的resnet模型）, get_cifar_densenet_model（针对cifar10的DenseNet模型）, get_gtsrb_resnet18（针对gtsrb的resnet模型）, get_MTFL_resnet18(针对人脸识别数据集的resnet18模型)


### 3.2 language
- 接口文件位置：auto_LiRPA_verifiy/language/verify.py
- 接口函数：verify()
- 调用示例：auto_LiRPA_verifiy/language/test.py
- 模型数据接口：支持sst情感分析数据集   
from auto_LiRPA_verifiy.language.data_utils import get_sst_data（获取sst数据）
from auto_LiRPA_verifiy.language.lstm import get_lstm_demo_model（获取lstm预训练模型）,
from auto_LiRPA_verifiy.language.Transformer.Transformer import get_transformer_model（获取Transformer预训练模型）


## 4 数据集
### 4.1 SST-2 情感分析数据集
- cd dataset/auto_LiRPA_dataset/data
- mkdir sst
- cd sst
- wget http://download.huan-zhang.com/datasets/language/data_language.tar.gz
- tar xvf data_language.tar.gz 
- 数据集介绍：SST-2(The Stanford Sentiment Treebank，斯坦福情感树库)，该数据集包含电影评论中的句子和它们情感的人类注释，是针对给定句子的情感倾向二分类数据集，类别分为两类正面情感（positive，样本标签对应为1）和负面情感（negative，样本标签对应为0）。该数据集包含了67, 350个训练样本，1, 821个测试样本。

### 4.2 MNIST 数据集
- 调用get_mnist_data()函数自动下载
- 数据集介绍：MNIST数据集(Mixed National Institute of Standards and Technology database)是美国国家标准与技术研究院收集整理的大型手写数字数据库，该数据集由10个类别的共计70,000个28*28像素的灰度图像组成，每个类有7,000个图像。

### 4.3 CIFAR-10 数据集
- cd dataset/auto_LiRPA_dataset/data
- wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- 数据集介绍：CIFAR-10数据集由CIFAR(Candian Institute For Advanced Research) 收集整理的一个用于机器学习和图像识别问题的数据集。该数据集由10个类别的共计60,000个32x32彩色图像组成，每个类有6000个图像。

### 4.4 GTSRB 数据集
- [GTSTB](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?select=Train) 页面下载achieve.zip压缩包
- 解压到 dataset/auto_LiRPA_dataset/data/gtsrb目录下
- 数据集介绍：GTSRB（German Traffic Sign Recognition Benchmark）是德国交通标志数据集，该数据集数据集包由43个类别的共计51,839幅像素不等的采自真实交通环境下的交通标志图像组成，其中训练和测试图像分别为39,209和12,630幅。

### 4.5 Multi-Task Facial Landmark (MTFL)数据集，（人脸识别数据集，识别男女性别）
- cd dataset/auto_LiRPA_dataset/data
- mkdir MTFL
- cd MTFL
- wget http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip
- unzip MTFL.zip
- 数据集介绍：该数据集由来自网络的12995张像素大小不等的真实人脸图片组成，该数据集标注了每张图片的五官坐标，性别、是否微笑、是否带眼睛，以及人脸朝向等信息。

## 5 Acknowledgments

- Code refers to [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA).

## 6 致谢

特别感谢
------
- [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) 为项目提供了重要的功能。
 
感谢所有贡献者的帮助，以及所有在这个项目中提出问题和建议的人们。
<br><br>
-- 2023.02.17
