# AI-platform
 ## 2030人工智能重大项目课题一子系统框架

## 运行：python main.py

 **在此之前，需要具备web前后端基本交互流程，ajax传参，flask基本运行原理等知识**

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
view：后台与前端的接口，每个python文件为一个蓝图，在flask生成时需要进行注册。api.py里面是之前我写的几个用到的，可以参考。  
为了方便，前端传参的js我暂时也写在了html文件里（最底部），采用ajax的传参框架，可以参考。

PS： 项目暂时没有配置工程上的如log、redis之类，后面有需要再考虑   


