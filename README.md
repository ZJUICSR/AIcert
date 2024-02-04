# 人工智能安全理论及验证平台
## 导读
随着人工智能技术的突飞猛进，它越来越多地参与到我们的日常生活和工作中，大幅提升了生产效率，丰富了我们的日常体验，并推动了不同行业间的紧密合作。因此在工业界和学术界，追求更高级的人工智能技术成为了一个持续的热点。然而，随着人工智能技术的广泛应用，其潜在的安全漏洞也逐渐暴露出来，这些漏洞可能对用户和整个系统造成严重威胁。例如，通过向原始输入数据中注入精心设计的噪声，可能导致模型做出错误决策，如在自动驾驶系统中，造成车辆无法正确识别交通标志。攻击者还可能在AI模型中植入后门，从而在满足一定条件时触发恶意功能，比如使得安全门禁系统失效，无法排除未授权人员。

即便是近期飞速发展的人工智能大模型，也未能摆脱被恶意利用的危险，例如在引导下进行越狱行为，甚至宣称能够设计出侵入银行系统的恶意软件。目前流行的人工智能大模型，无论是OpenAI的ChatGPT、Google的Bard、Meta的LLaMA-2，还是百度的文心一言，以及阿里的通义千问等，都不可避免地面临着严重的安全隐患。这些挑战提醒我们，在享受人工智能带来的便捷与效率之时，更应加强警惕，确保采取适当的安全措施，以保障AI技术的安全稳定运行。
![image1.png](https://s2.loli.net/2024/02/04/DZoetFhBEN5xAKr.jpg)
![image2.png](https://s2.loli.net/2024/02/04/xt3INsnKmledbku.jpg)

人工智能系统面临的安全威胁原因是多方面的。首先是系统本身的复杂性与不透明性，当前主流的人工智能系统大多构建于深度神经网络之上,网络结构复杂,训练过程是个黑盒操作,模型的决策逻辑对用户不透明。这为攻击者提供了可乘之机,难以检测或防范所有可能的攻击方式。其次数据和模型质量存在问题，如果训练数据有偏差、标签错误,或者样本量不足,都可能导致模型在实际部署后产生意外判断失误。此外,训练好的模型本身也可能因算法缺陷而存在固有安全漏洞。此外,系统部署和运维过程的安全漏洞也是重要因素，复杂的人工智能系统在工程实现和产品部署中,其软硬件基础设施都可能带来漏洞。最后,在部分应用场景下,人工智能系统还面临着数据隐私泄露的风险。总体而言,人工智能安全问题囊括了技术复杂性、系统脆弱性、应用场景限制等多方面因素。需要系统性解决方案来应对日益严峻的安全威胁,保证人工智能的健康与可控发展。
## 人工智能安全理论及验证平台VoAI

浙江大学区块链与数据安全全国重点实验室长期深耕人工智能领域，致力于AI安全领域研究工作，于近日正式发布并开源**人工智能安全理论及验证平台——VoAI平台**。该平台旨在解决当前AI安全面临的三大问题：AI模型结构复杂、不透明、缺乏理论可解释性；AI系统链路长、安全防护弱、缺乏可信测试技术；AI系统鲁棒性差、缺乏可拓展性、场景复杂多变。为应对上述问题，团队面向大规模复杂智能系统的一体化安全保障技术体系与验证技术、智能系统全链路安全可信与适配技术和模型脆弱性感知与安全增强的主动防御技术，从理论分析、开发测试、运行环境优化、防御增强、性能验证五个维度出发研发了VoAI平台。平台旨在对AI系统的全链路和全生命周期威胁进行感知与安全评估、提供安全防御策略，具备**全链路威胁感知**、**多维度安全评估**和**动态安全防御**三大特点，显著提升了人工智能系统的威胁监测、预警和响应能力，减少了由AI系统脆弱性引起的安全风险，为人工智能产业的安全发展提供了强大的动力。


### 全链路威胁感知：打破传统安全边界的限制，实现全链路覆盖

在传统观念中，AI安全的防护主要集中在数据安全、模型安全和应用安全三个方面，而VoAI打破了这一传统观念，实现了对全链路的覆盖。VoAI的关注不仅限于数据和模型，而是将安全防护扩展至AI系统的各个环节，包括数据、模型、算法、开发框架以及操作系统。通过全链路威胁感知，VoAI能够更准确地评估AI系统的安全状况，及时发现并消除潜在的安全隐患，从而增强AI系统的安全性和可靠性。同时，这种感知能力也为AI系统的持续优化和改进提供了坚实的支持，助力AI系统更有效地应对日益复杂多变的安全挑战。
![image3.png](https://s2.loli.net/2024/02/04/8Sj7cMZLY9wfW1i.png)
*  AI数据面临多重安全威胁：训练数据可能被污染，例如在分类模型训练过程中，攻击者可能向训练数据中恶意混入毒化数据，导致模型产生分类错误。同时，应用输入数据也可能受到篡改，例如攻击者使用对抗样本欺骗人工智能系统，以导致系统做出错误的判断。为应对这些威胁，VoAI平台具备对上传数据集进行异常检测和公平性评估的能力，有效评估数据集质量，并提供异常数据修复与公平性增强的功能，从而进一步提升数据集的质量。
![image4.png](https://s2.loli.net/2024/02/04/kWzPv9jaNEY6qdJ.png)
*   AI模型脆弱性：AI模型的脆弱性表现为微小的数据错误就会使系统发生故障。例如，在图像识别中，向图片中加入肉眼无法察觉的微小扰动，就可能会导致AI系统判断错误。VoAI平台采用了基于注意力机制和模型预测置信度信息的双通道对抗样本检测技术，实现了对**60+**种对抗攻击的高效感知。此外，VoAI平台还集成了**6种**后门攻击检测算法，有效识别后门攻击。
![image5.png](https://s2.loli.net/2024/02/04/W6D4Cw1HtZLK9V3.png)
*   AI算法“黑盒”风险：随着神经网络模型的深度增加和参数空间的复杂化，图像、语音和文本识别等任务性能得到了显著提升，但同时也导致了模型内部逻辑的难以理解，这对医疗和自动驾驶等高可靠性支撑场景构成重大隐患。VoAI平台通过攻击机理分析深入研究各类对抗性攻击的攻击样本生成与作用机理，解析其误导模型决策的内在原因，并使用直观、易于理解的解释方法展示风险。
![image6.png](https://s2.loli.net/2024/02/04/7iKJ6rY8LfGDMut.png)
*   AI开发框架与操作系统漏洞频发：尽管AI开发框架可以帮助开发者高效、快速地构建模型并进行训练和推理，然而，在这种便捷高效的表象之下，AI开发框架也存在多种安全漏洞，可能导致数据泄露、模型篡改和系统后门等安全问题。VoAI平台通过对Windows、Ubuntu、CentOS这**3种**操作系统以及TensorFlow、PyTorch、百度飞桨、CNTK、Theano这**5种**主流开发框架的安全度量，能够在这些主流开发框架上实现漏洞检测的高准确率、高效率和高通用性，检测误报率<3%，优于国内外现有技术的检测结果。
![image7.png](https://s2.loli.net/2024/02/04/Q95Gm8lFML1dHKO.png)
## 多维度安全评估：系统多维评估，实现全方位安全评测

针对AI系统模型安全、数据安全和运行环境安全等方面的安全挑战，VoAI平台从鲁棒性、公平性、完整性、可用性、可解释性、可验证性六大维度对AI系统进行评估，全方位审视AI系统的安全性。六大维度安全性评测是人工智能系统多维度综合评估的创新性探索，帮助工业界与学界更好的评估AI系统，促进AI规模化落地。
![image8.png](https://s2.loli.net/2024/02/04/wAYJePNsF49Dy5V.png)
*   **鲁棒性**用于评估AI模型受到扰动与攻击时能否保持稳定运行。VoAI平台已集成了60+种主流的对抗样本生成算法，其中包括50种白盒对抗攻击方法和11种黑盒攻击方法，以全面评估AI模型的抗攻击能力。此外，VoAI平台还引入了6种后门攻击，旨在挖掘模型更多的潜在漏洞。
![image9.png](https://s2.loli.net/2024/02/04/qF9OCBG5hpjKSJA.png)
*   **公平性**用于评估人工智能系统中的数据与模型是否存在偏见与歧视性决策。VoAI平台实现了全面且高可用性的公平性评估与提升功能，拥有超过30种公平性评估指标，从公平性准则的角度来说，覆盖了群体公平性和个体公平性；从评估对象的角度来说，覆盖了数据集公平性与模型公平性，能够实现精准、全面的公平性评估与公平性优化。
![image10.png](https://s2.loli.net/2024/02/04/uI6nLgQUxiZsSDE.png)
*   **完整性**用于评估数据在生命周期内是否遭到破坏，如格式异常数据、毒化数据等。VoAI平台支持表格、文本、图像3种模态的数据类型，可检测离群值、编码格式异常以及毒化标签异常等3种高频异常，并对其中的对可修复数据进行修复，不可修复数据进行弃置，实现对数据的完整性评估及完整性提升。
![image11.png](https://s2.loli.net/2024/02/04/7GOThtDlaC18UzX.png)
*   **可用性**用于评估开发框架和操作系统是否存在不兼容问题和安全漏洞，以确保软件层和应用层安全可用。VoAI平台的开发环境分析功能不仅能够对系统架构信息、依赖库版本、以及AI开发框架的依赖与版本等关键信息进行分析，还能发现系统中潜在的漏洞问题。在框架适配功能方面，该平台能够在框架功能适配模块中，对用户指定的开发框架进行一系列的分析和比对操作，包括对开发框架版本、依赖库等信息的检测。通过此功能，系统能够评估执行环境是否存在安全风险，并生成报告清晰说明潜在的安全风险。
![image12.png](https://s2.loli.net/2024/02/04/V8nQiBRJ2XqPI36.png)
*   **可解释性**是指对模型决策过程的解释分析，旨在使人类用户能够理解、信任和解释机器学习模型的工作原理及推理基础。VoAI平台集成8种可解释性方法，包括征归因可视化、数据分布降维可视化、模型内部特征分析可视化等。这些技术的应用使得VoAI平台能够深入分析模型的决策过程，并通过可视化的手段揭示模型在决策时的注意力焦点。如图所示，该图展示了水蛇图像及其对抗样本图像在不同可解释性算法下的显著性图。观察结果表明，模型的注意力焦点在对抗样本的影响下发生了显著变化，从蛇身转移至地面。通过这种方法，研究人员和开发者可以更好地理解模型的行为，并采取措施来提高模型的鲁棒性和可靠性。
![image13.png](https://s2.loli.net/2024/02/04/CJlk1IEqQH5FAZy.png)
*   **可验证性**通过数学工具对算法模型的潜在行为空间进行理论分析，采用形式化验证方法有助于理解无限的行为模式，从而实现主流AI系统安全特性的快速验证。深度度学习模型在任意扰动的作用下可能出现输出不符合预期的情况，平台的形式化验证通过模型特征安全性验证、模型一致性验证、输出空间可达性验证，能够直观的展示模型在特定应用场景下的输出是否符合预期。
![image14.png](https://s2.loli.net/2024/02/04/cjtWFrymTQnZo19.png)
## 动态安全防御：打破孤立保护机制，实现群智增强防御

为了应对现有AI模型保护机制在孤立性、覆盖范围有限和防御策略固定性等方面的挑战，VoAI平台基于感知与评估，提出了一种创新的动态安全防御增强方法，该方法突破攻击行为动态感知、群智化防御和动态安全防御策略自动生成等核心关键技术。通过这些技术突破，VoAI平台能够实现基于群体智能的动态安全防御增强，这种防御方法将攻防博弈推演与群智化防御相结合，不仅增强了AI系统的安全性，而且提高了防御策略的灵活性和自适应性，确保了AI系统在面对不断变化的安全威胁时能够保持鲁棒性和安全性。
![image15.png](https://s2.loli.net/2024/02/04/Ih5SBpF1Dr6w4lE.png)
*   **攻防博弈推演：**VoAI平台内置的纳什博弈、演化博弈等5种博弈算法，集成60+种对抗样本攻击方法和防御方法，构建了一套完整的攻防博弈推演系统。该平台通过模拟真实场景中的红方和蓝方，动态计算在不同给定对手策略下的收益矩阵。在攻防推演过程中，双方根据对手可能的策略和行动调整自身策略，以追求最优策略，实现自主动态防御增强机制。这种基于博弈理论和对抗样本的综合推演方法，为VoAI平台提供了深度的安全性评估和强化机制，有效应对复杂多变的对抗攻击，为系统的鲁棒性和安全性奠定了坚实基础。

*   **群智化防御增强：**VoAI平台内置了基于群智协同防御和自适应多策略防御等**5种**群智化防御方法，构建了特征散射法、随机平滑法和对抗训练等对抗防御知识库。通过群智防御策略，VoAI平台强化了AI模型在面对多种攻击下的鲁棒性和安全性。该平台提出了一种集成防御算法，运用多种攻击算法生成的对抗样本进行对抗训练，形成高鲁棒性的模型，显著提升了模型对各种对抗样本生成算法的抵抗力。群智化防御增强方法以集成防御机制为核心，有效确保系统能够应对不断演进的对抗攻击，为平台提供了坚实的安全保障。

### 平台对比
人工智能安全问题备受业界瞩目，在此领域涌现出大量相关产出，仅在Github平台上就有1600余个相关项目。为深入比较，我们挑选了几个主流人工智能安全评测平台与VoAI进行对比，具体对比结果如下表所示。在研究范围方面，VoAI的覆盖范围更广，不仅加强了对AI框架的安全防护，还拓展至整个AI系统的全链路安全；而在功能上，相较于其他平台，VoAI呈现更为深入的特性，能够从多个角度对AI系统的安全性进行评测，并提供相应的加固策略。
![image16.png](https://s2.loli.net/2024/02/04/GoSsFYhI3nKUyaZ.jpg)

## 保障大模型安全

2022年3月以来，以ChatGPT、Bard为代表的大模型席卷全球，引发了人们对于通用人工智能的广泛关注，大模型的快速发展深刻地改变了人们的生活和生产方式，例如自动会议总结、报告生成和代码补全等。虽然大模型掀起了一场重要的技术变革，但目前大模型仍然存在严重的安全性问题。

尽管大模型在自然语言处理任务中展现出显著的性能优势，却仍面临传统AI算法所未曾遇到的安全性与鲁棒性挑战。这些模型处理开放世界中复杂问题的能力尚不完备。研究表明，大模型对异常数据的鲁棒性不足，易受输入微小变化的影响。例如，提示词顺序或拼写的修改，或文本中的噪声添加，均可能引发模型输出错误或不稳定结果。这归因于模型可能过分依赖预训练数据，缺少对特定任务或领域的自适应能力。
![image17.png](https://s2.loli.net/2024/02/04/3j5kB1LKR9fzAyw.jpg)
为应对上述挑战，VoAI平台从字符级别、单词级别、句子级别等3个攻击维度，采用TextBugger、TextFooler等4种对抗攻击方法对17个主流开源大型语言模型在57个主题任务上进行了全面的鲁棒性评估，模型规模超亿级。评估结果显示，这些大模型的平均准确率下降了8.8%，反映出大模型在不同程度上存在鲁棒性不足问题，对微小输入变化的敏感性较强。展望未来，研究团队计划进一步完善大模型多维度的安全性评估能力，特别是针对大模型的越狱风险、模型幻觉和提示语注入等关键安全问题，进行更深入的分析和评估。


## 结语

随着人工智能技术逐渐渗透到社会的各个领域，从医疗健康、金融服务到智能制造和自动驾驶，其安全性正受到前所未有的关注。在这个背景下，人工智能安全理论及验证平台VoAI的出现具有重要意义。它不仅针对深度学习模型，也对涉及大量参数和复杂结构的大模型进行了安全评测，提供了一系列工具和框架，以识别和评估这些模型可能存在的安全漏洞。通过这种方式，VoAI使得研究人员和开发者能够在模型部署前预先检测到潜在的风险，从而采取必要的补救措施。VoAI作为一个综合性的解决方案，有效地弥补了现有安全评估工具在面对复杂AI系统时的不足，使得安全评测更加系统化和标准化，为AI安全领域的研究和应用提供一个坚实和可靠的基础。

区块链与数据安全全国重点实验室，依托浙江大学，于2022年11月正式获得国家科技部批准成立。实验室聚焦区块链与数据安全国际科技前沿，以实现高水平科技自立自强和打造具有世界一流的战略科技力量为己任，围绕产学研一体融合，开展系统性创新性科技攻关。实验室的研究方向主要包括区块链技术与平台、区块链监管监测、智能合约与分布式软件、数据要素安全与隐私计算、AI数据安全与认知对抗、AI原生数据处理系统、网络数据治理、智能网联车数据安全、可信数据存储与计算技术等。
浙江大学区块链与数据安全全国重点实验室人工智能数据安全团队由常务副主任、计算机学院院长任奎教授牵头，团队牵头承担了科技部科技创新2030重大项目中人工智能安全领域首个重大项目“人工智能安全理论与验证平台”，项目团队由浙江大学、武汉大学、西安交通大学、南京航空航天大学、西北工业大学、淘宝（中国）软件有限公司、山东省计算中心（国家超级计算济南中心）、中国人民公安大学、湖南四方天箭信息科技有限公司联合组建。团队研究成果覆盖人工智能系统硬件、操作系统、软件、模型、数据、安全策略等多个层次，已发表计算机安全、人工智能领域国际顶会论文100余篇，其中多篇获得最佳论文奖。

为了促进科研成果的共享和行业的快速发展，并推动研究成果的广泛应用。我们团队将人工智能安全理论与验证平台VoAI的代码公开，供全球研究人员和开发者使用和参与改进。感兴趣的个人和组织可以通过下方提供的Gitee开源地址访问和下载相关代码，同时也可以通过提供的平台链接直接体验VoAI平台的功能和性能。

# 平台部署
1. 环境配置 python>=3.7, <3.10
	```
	pip install -r requirement.txt
	```

2. 运行：python main.py  --port 端口
3. 使用Google浏览器打开界面


 ### 项目基本架构介绍：

 config: web运行时的设置（不需要管）  
 function：后台的总接口，里面存放调用各个算法的函数
 dataset： 需要的数据集缓存目录，如mnist\cifar10等    
 logs: 日志存储文件，暂时没做  
 model： 暂时放了预训练模型，后面应该会加入一些构建预训练模型的代码  
 output： 暂时没用，后面会作为输出的缓存  
 utils:工具库，暂时没用，不确定需要哪些共同算法  
 web：前端所有文件  

### web目录：

static 为静态目录，其下的所有文件在整个flask框架启动后，前端都可以访问到，flask也可以自行指定static目录  
templates：存储所有的前端html页面  
view：后台与前端的接口，每个python文件为一个蓝图，在flask生成时需要进行注册。

# 基于[audo_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)的模型形式化验证接口

## 1. 环境安装

### 1.1 audo_LiRPA安装

1. 进入audo_LiRPA目录执行命令：

    cd auto_LiRPA\auto_LiRPA  
    python setup.py install  

2. python依赖库安装：  
   pip install -r requirements.txt
   


### 1.2 第三方包约定

*输出文件的地址，在配置文件中写出，一般为绝对目录*

需包含文件以及说明如下：
1. requirements.txt - python依赖包文件


### 1.2 接口信息
1. params.json - 输入参数

```json = 
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

```json = 
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
cd third_party.auto_LiRPA_verifiy/language/  
mkdir model  
cd model/  
wget http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/trained/ckpt_lstm  
wget http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/trained/ckpt_transformer  



## 3. 接口位置
### 3.1 vision
- 接口文件位置：/auto_LiRPA/third_party/auto_LiRPA_verifiy/vision/verify.py
- 接口函数：verify()
- 调用示例：/auto_LiRPA/third_party/auto_LiRPA_verifiy/vision/test.py
- 模型数据接口：支持mnist，cifar，gtsrb数据集   
from auto_LiRPA_verifiy import get_mnist_data(获取mnist数据), get_cifar_data(获取cifar数据), get_gtsrb_data(获取gtsrb交通数据), get_MTFL_data(获取人脸识别数据集)  
from auto_LiRPA_verifiy import get_mnist_cnn_model（针对mnist的CNN模型）, get_cifar_resnet18（针对cifar的resnet模型）, get_cifar_densenet_model（针对cifar10的DenseNet模型）, get_gtsrb_resnet18（针对gtsrb的resnet模型）, get_MTFL_resnet18(针对人脸识别数据集的resnet18模型)


### 3.2 language
- 接口文件位置：/auto_LiRPA/third_party/auto_LiRPA_verifiy/language/verify.py
- 接口函数：verify()
- 调用示例：/auto_LiRPA/third_party/auto_LiRPA_verifiy/language/test.py
- 模型数据接口：支持sst情感分析数据集   
from auto_LiRPA_verifiy import get_sst_data（获取sst数据）
from auto_LiRPA_verifiy import get_lstm_demo_model（获取lstm预训练模型）, get_transformer_model（获取Transformer预训练模型）


## 4 数据集
### 4.1 SST-2 情感分析数据集
- cd auto_LiRPA/third_party/auto_LiRPA_verifiy/language/data
- wget http://download.huan-zhang.com/datasets/language/data_language.tar.gz
- tar xvf data_language.tar.gz 
- 数据集介绍：SST-2(The Stanford Sentiment Treebank，斯坦福情感树库)，该数据集包含电影评论中的句子和它们情感的人类注释，是针对给定句子的情感倾向二分类数据集，类别分为两类正面情感（positive，样本标签对应为1）和负面情感（negative，样本标签对应为0）。该数据集包含了67, 350个训练样本，1, 821个测试样本。

### 4.2 MNIST 数据集
- 调用get_mnist_data()函数自动下载
- 数据集介绍：MNIST数据集(Mixed National Institute of Standards and Technology database)是美国国家标准与技术研究院收集整理的大型手写数字数据库，该数据集由10个类别的共计70,000个28*28像素的灰度图像组成，每个类有7,000个图像。

### 4.3 CIFAR-10 数据集
- cd auto_LiRPA/third_party/auto_LiRPA_verifiy/vision/data
- wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- 数据集介绍：CIFAR-10数据集由CIFAR(Candian Institute For Advanced Research) 收集整理的一个用于机器学习和图像识别问题的数据集。该数据集由10个类别的共计60,000个32x32彩色图像组成，每个类有6000个图像。

### 4.4 GTSRB 数据集
- [GTSTB](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?select=Train) 页面下载achieve.zip压缩包
- 解压到 auto_LiRPA/third_party/auto_LiRPA_verifiy/vision/data/gtsrb目录下
- 数据集介绍：GTSRB（German Traffic Sign Recognition Benchmark）是德国交通标志数据集，该数据集数据集包由43个类别的共计51,839幅像素不等的采自真实交通环境下的交通标志图像组成，其中训练和测试图像分别为39,209和12,630幅。

### 4.5 Multi-Task Facial Landmark (MTFL)数据集，（人脸识别数据集，识别男女性别）
- cd auto_LiRPA/third_party/auto_LiRPA_verifiy/vision/data
- mkdir MTFL
- cd MTFL
- wget http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip
- unzip MTFL.zip
- 数据集介绍：该数据集由来自网络的12995张像素大小不等的真实人脸图片组成，该数据集标注了每张图片的五官坐标，性别、是否微笑、是否带眼睛，以及人脸朝向等信息。

## 5 Acknowledgments

- Code refers to [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA).



