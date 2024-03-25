# 人工智能安全理论及验证平台 AIcert

## 平台部署
1. 环境配置 python>=3.7, <3.10
	```
	pip install -r requirement.txt
	```

2. 运行：python main.py  --port 端口
3. 使用Google浏览器打开界面


浙江大学人工智能团队长期深耕人工智能领域，致力于发展AI安全，于近日正式发布并开源**人工智能安全理论及验证平台VoAI**平台**。**平台旨在对AI系统的全链路和全生命周期威胁进行感知与安全评估、提供安全防御策略，具备**全链路威胁感知**、**多维度安全评估**和**动态安全防御**三大特点，显著提升了人工智能系统的威胁监测、预警和响应能力，减少了由AI系统脆弱性引起的安全风险，为人工智能产业的安全发展提供了强大的动力。

### 全链路威胁感知：打破传统安全边界的限制，实现全链路覆盖

在传统观念中，AI安全边界的防护主要集中在数据安全、模型安全和应用安全三个方面。然而，VoAI打破了这一传统观念，实现了对全链路的覆盖。它不仅关注数据和模型，还将安全防护拓展至AI系统的每一个环节，包括智能数据、模型算法、开发框架以及操作系统等。通过全链路的威胁感知，我们能够更准确地把握AI系统的安全状态，及时发现并解决潜在的安全隐患，从而增强AI系统的安全性和可靠性。这种感知能力也为AI系统的优化和改进提供了强有力的支持，帮助我们更好地应对日益复杂多变的安全威胁。

static 为静态目录，其下的所有文件在整个flask框架启动后，前端都可以访问到，flask也可以自行指定static目录  
templates：存储所有的前端html页面  
view：后台与前端的接口，每个python文件为一个蓝图，在flask生成时需要进行注册。

## 基于[audo_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)的模型形式化验证接口

### 1. 环境安装

#### 1.1 audo_LiRPA安装

1. 进入audo_LiRPA目录执行命令：

    cd auto_LiRPA\auto_LiRPA  
    python setup.py install  

2. python依赖库安装：  
   pip install -r requirements.txt
   


#### 1.2 第三方包约定

*输出文件的地址，在配置文件中写出，一般为绝对目录*

需包含文件以及说明如下：
1. requirements.txt - python依赖包文件

### 2. Element Definition

#### 2.1 训练数据类型：
- image(mnist, cifar10， gtsrb, MTFL)
- language(SST-2)

#### 2.2 模型类型：
- CNN、ResNet、DenseNet;
- LSTM、Transformer;
下载LSTM和Transformer预训练模型：  
cd third_party.auto_LiRPA_verifiy/language/  
mkdir model  
cd model/  
wget http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/trained/ckpt_lstm  
wget http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/trained/ckpt_transformer  



### 3. 接口位置
#### 3.1 vision
- 接口文件位置：/auto_LiRPA/third_party/auto_LiRPA_verifiy/vision/verify.py
- 接口函数：verify()
- 调用示例：/auto_LiRPA/third_party/auto_LiRPA_verifiy/vision/test.py
- 模型数据接口：支持mnist，cifar，gtsrb数据集   
from auto_LiRPA_verifiy import get_mnist_data(获取mnist数据), get_cifar_data(获取cifar数据), get_gtsrb_data(获取gtsrb交通数据), get_MTFL_data(获取人脸识别数据集)  
from auto_LiRPA_verifiy import get_mnist_cnn_model（针对mnist的CNN模型）, get_cifar_resnet18（针对cifar的resnet模型）, get_cifar_densenet_model（针对cifar10的DenseNet模型）, get_gtsrb_resnet18（针对gtsrb的resnet模型）, get_MTFL_resnet18(针对人脸识别数据集的resnet18模型)


#### 3.2 language
- 接口文件位置：/auto_LiRPA/third_party/auto_LiRPA_verifiy/language/verify.py
- 接口函数：verify()
- 调用示例：/auto_LiRPA/third_party/auto_LiRPA_verifiy/language/test.py
- 模型数据接口：支持sst情感分析数据集   
from auto_LiRPA_verifiy import get_sst_data（获取sst数据）
from auto_LiRPA_verifiy import get_lstm_demo_model（获取lstm预训练模型）, get_transformer_model（获取Transformer预训练模型）


### 4 数据集
#### 4.1 SST-2 情感分析数据集
- cd auto_LiRPA/third_party/auto_LiRPA_verifiy/language/data
- wget http://download.huan-zhang.com/datasets/language/data_language.tar.gz
- tar xvf data_language.tar.gz 
- 数据集介绍：SST-2(The Stanford Sentiment Treebank，斯坦福情感树库)，该数据集包含电影评论中的句子和它们情感的人类注释，是针对给定句子的情感倾向二分类数据集，类别分为两类正面情感（positive，样本标签对应为1）和负面情感（negative，样本标签对应为0）。该数据集包含了67, 350个训练样本，1, 821个测试样本。

#### 4.2 MNIST 数据集
- 调用get_mnist_data()函数自动下载
- 数据集介绍：MNIST数据集(Mixed National Institute of Standards and Technology database)是美国国家标准与技术研究院收集整理的大型手写数字数据库，该数据集由10个类别的共计70,000个28*28像素的灰度图像组成，每个类有7,000个图像。

#### 4.3 CIFAR-10 数据集
- cd auto_LiRPA/third_party/auto_LiRPA_verifiy/vision/data
- wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- 数据集介绍：CIFAR-10数据集由CIFAR(Candian Institute For Advanced Research) 收集整理的一个用于机器学习和图像识别问题的数据集。该数据集由10个类别的共计60,000个32x32彩色图像组成，每个类有6000个图像。

#### 4.4 GTSRB 数据集
- [GTSTB](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?select=Train) 页面下载achieve.zip压缩包
- 解压到 auto_LiRPA/third_party/auto_LiRPA_verifiy/vision/data/gtsrb目录下
- 数据集介绍：GTSRB（German Traffic Sign Recognition Benchmark）是德国交通标志数据集，该数据集数据集包由43个类别的共计51,839幅像素不等的采自真实交通环境下的交通标志图像组成，其中训练和测试图像分别为39,209和12,630幅。

#### 4.5 Multi-Task Facial Landmark (MTFL)数据集，（人脸识别数据集，识别男女性别）
- cd auto_LiRPA/third_party/auto_LiRPA_verifiy/vision/data
- mkdir MTFL
- cd MTFL
- wget http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip
- unzip MTFL.zip
- 数据集介绍：该数据集由来自网络的12995张像素大小不等的真实人脸图片组成，该数据集标注了每张图片的五官坐标，性别、是否微笑、是否带眼睛，以及人脸朝向等信息。

### 5 Acknowledgments

- Code refers to [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA).




![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/collab/jP2lR4zeY27Bq8g5/23bf790e-115d-45e3-8cd9-5d7dab2ce677.png "AI算法“黑盒”")

*   AI框架与操作系统漏洞频发：AI框架可以帮助开发者高效、快速地构建网络模型进行训练和推理，具有高效、便捷的特性。然而，在便捷高效的表象之下，人工智能框架也存在许多问题与风险。目前，平台支持Windows、Ubuntu、CentOS这三种操作系统以及TensorFlow、PyTorch、百度飞桨、CNTK、Theano这五种主流开发框架的安全度量。该技术能够在这些主流开发框架上实现高准确率、高效率、高通用性的漏洞检测，其检测误报率低于3%，这一结果优于国内外现有技术的检测结果。
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ABmOorDZDX9eqawZ/img/2633c160-de5a-4a4d-ba54-0cfc74b7bb38.png?x-oss-process=image/crop,x_7,y_10,w_806,h_472 "框架开发团队确认修复漏洞公告 ")

## 多维度安全评估：系统多维评估，实现全方位安全评测

针对AI系统模型鲁棒、数据安全和运行环境安全等方面的安全挑战，平台从鲁棒性、公平性、完整性、可用性、可解释性、可验证性六大维度对AI系统进行全面评估，全方位审视AI系统的安全性；六大维度安全性评测是人工智能系统多维度综合评估的创新性探索，帮助业界更好的评价人工智能系统，促进AI规模化落地。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ABmOorDZDX9eqawZ/img/680b65e5-566a-490c-a815-6d9f4b647da2.png)

*   **鲁棒性**是对智能模型受到扰动与攻击能否稳定运行的评估。平台已集成了60多种主流的对抗样本生成算法，其中包括50种经典白盒对抗攻击和11种黑盒攻击，以全面评估AI模型的抗攻击能力。此外还引入了6种后门攻击，旨在挖掘模型更多的潜在漏洞。以VoAI在开源模型ResNet上的攻击评测结果为例，使用CIFAR10数据集进行测试。结果显示60多种攻击算法中，一半以上算法的攻击成功率超过90%，即图像分类任务准确率大大降低。
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/pAgjegM24SGZZyBq/27c8f334ed134d86a4ca1f2878156bf31208.png "人工智能系统攻击结果")

*   **公平性**是对人工智能系统中的数据与模型是否存在偏见与歧视性决策的评估。平台实现了一个全面 且高可用的人工智能公平性评估与提升功能。该功能模块从公平性准则的角度覆盖了群体公平性和个体公平性；从评估对象的角度覆盖了数据集公平性与模型公平性。并针对两个对象提供了公平性评估和提升功能。平台拥有超过30种评估指标，能够实现精准的公平性评估与公平性优化治理。
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ABmOorDZDX9eqawZ/img/3da099a2-e15d-4c23-9489-bfb41803879f.png)

*   **完整性**是对数据在生命周期内是否遭到破坏的评估，如被添加格式异常数据、毒化数据等，为模型训练或部署提供数据保障。数据是系统的基础，平台支持表格、文本、图像3种模态的数据类型，可检测离群值、编码格式异常、毒化标签异常等4种频发异常。人工智能数据的完整性可以从根源上为人工智能系统安全提供保障，是人工智能系统正确决策和有效运行的根本。
    

*   **可用性**是对开发框架、操作系统进行兼容性匹配与漏洞检测，确保软件层和应用层安全可用。平台的开发环境分析功能，可以对不同操作系统下的系统架构信息、依赖库版本、AI开发框架依赖与版本等关键信息进行分析与记录，挖掘系统潜在的漏洞问题。框架适配功能，对于用户指定的开发框架，在框架功能适配模块中进行一系列的分析和比对操作。这包括对开发框架的版本、依赖库等信息进行核实，判断当前环境是否可以使用该框架功能。如果当前环境无法使用该框架功能，那么会生成一份报告，详细说明存在的问题和限制。
    

*   **可解释性**指能够清晰、透明地解释算法做出的决策或预测的过程。VoAI平台通过8种可解释性方法，如征归因可视化、数据分布降维可视化、模型内部特征分析可视化，深入分析攻击行为内在机理，并使用易于用户理解的方式可视化展示。特征归因可视化通过解释算法计算模型在正常样本和对抗样本上的显著图，并做可视化标注处理和展示，如下展示的是牧羊犬在不同解释算法下的显著图，可以看到对抗样本显著改变模型关注的特征区域，从狗脸移到了边缘位置，说明模型关注区域发生了偏移。
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ABmOorDZDX9eqawZ/img/9c4e802f-3035-4fdd-ad6e-ef960df0a4f8.png)

*   **可验证性**通过数学工具对算法模型的潜在行为空间进行理论分析，采用形式化验证方法有助于理解无限的行为模式，从而实现主流AI系统安全特性的快速验证。深度度学习模型在任意扰动的作用下可能出现输出不符合预期的情况，平台的形式化验证通过模型特征安全性验证、模型一致性验证、输出空间可达性验证，能够直观的展示模型在特定应用场景下的输出是否符合预期。
    

## 动态安全防御：打破孤立保护机制，实现群智增强防御

VoAI平台在感知与评估的基础上，针对AI模型保护机制设计孤立、覆盖环节有限、防御策略固定等特点，提出了群智增强的AI模型防御方法。突破防御智能体结构设计、智能体博弈机制、群体智能防御算法等关键技术，实现了基于群体智能的模型安全动态防御，即攻防博弈推演与群智增强防御。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ABmOorDZDX9eqawZ/img/140cd90f-7904-4237-a608-ff3ba1cb5017.png)

*   攻防博弈推演：团队在对抗样本攻防博弈领域，将攻防对抗视为一个双人博弈问题，其中防御算法和攻击算法分别扮演双方，采用防御算法的收益函数作为评估指标，生成一个收益矩阵从而确定特定场景下的最优攻防策略，实现攻防博弈推演功能。功能模块中集成60多种对抗样本生成算法作为攻击策略选择，采用不同的攻击算法进行对抗训练作为防御策略。通过这种非对称能力的群体博弈算法，最终得到攻守双方的收益矩阵，选出特定场景下的最优攻防策略。
    

*   平台群智增强防御提供了鲁棒性训练和集成防御功能。鲁棒性训练通过特征散射、异常感知、随机平滑、对抗训练等方法提升深度学习模型和图神经网络的稳定性和准确性。鲁棒性训练的目的是使模型在面对对抗样本等不同的场景下都能表现良好。然而使用一种的对抗样本进行对抗训练，在应对其他类型的攻击算法方面效果却不尽如人意。针对这个问题，平台实现了一种基于多种攻击算法的集成防御算法，通过使用多种攻击算法生成的对抗样本分别进行对抗训练，综合多个训练结果组成最终分类模型，从而提升模型应对不同类型对抗样本生成算法的鲁棒性。
    

## 为大模型的发展保驾护航

人工智能已迈入大模型时代，大模型为大量下游领域提供支撑，应用广泛。国内大模型的发展也相当迅速，推出许多大模型应用，如百度的文心一言，复旦大学的MOSS，阿里的通义千问等，竞争激烈。

相对于传统AI算法，大模型具有新的特性和风险。大模型数据来源多样，但同样面临更大的攻击安全风险。大模型具有智能涌现特性，应用场景剧增，赋能下游应用技术升级，但同质化严重，大模型的任何改变都会影响整个社区，包括其存在的安全缺陷。

大模型的各方面性能普遍优于普通的AI模型，但仍面临传统的安全问题，例如隐私泄露，对抗攻击，歧视等安全风险。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ABmOorDZDX9eqawZ/img/28f2946b-5ac9-49f0-9686-bb8105b01b85.png)

VoAI平台紧跟时代步伐，构建起大模型安全测评能力。针对多项中英文任务，使用BoolQ和MMLU对17个主流开源大模型进行鲁棒性评测，大模型平均准确率下降达8.80%，可以看出开源的大模型存在不同程度的鲁棒性不足问题，轻微的扰动对大模型性能的影响较大。未来，团队还将更进一步实现覆盖大模型全生命周期安全的测评能力建设，重点对大模型后门攻击威胁、毒化攻击威胁、输出内容鲁棒性、越狱安全风险、软件框架安全性等大模型安全问题进行评测。

## 结语

人工智能安全理论及验证平台VoAI提供了一个深度学习模型和大模型安全评测的综合型解决方案，填补了现有工具的空白，为AI安全提供更为全面、有效地保障。

人工智能安全团队负责人为浙江大学区块链与数据安全全国重点实验室常务副主任、计算机学院院长任奎教授，由浙江大学、武汉大学、西安交通大学、南京航空航天大学、西北工业大学、淘宝（中国）软件有限公司、山东省计算中心（国家超级计算济南中心）、中国人民公安大学、湖南四方天箭信息科技有限公司联合组建。团队包含多名教授和几十名博士、硕士研究生，研究成果覆盖人工智能系统各个层次，包含硬件、操作系统、软件、攻击、防御，已发表计算机安全、人工智能顶会论文100余篇，其中多篇获得最佳论文奖。团队积极响应国家号召，抢占人工智能技术制高点，妥善应对人工智能带来的安全新问题，实现关键技术自主可控。