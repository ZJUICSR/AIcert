接口函数在下列代码中：

```
class ConcolicShow(object):
	def show_results(model, data, model_name: str, data_name: str, norm: str, basepath: str, out_path:str,Times:int=0):
	...
```

该函数返回一个json, 如果需要修改或增加返回的内容，可以修来里面的内容。

接口调用Demo见'______init__.py____', 其中必须有的参数为：

```
	params={
		"concolic_dataset":"cifar10",
		"concolic_model":'vgg',
		"norm":'linf',
		"basepath":'/data/jwp/codes/preTestGen23/argp/argp/third_party/preprocessing/concolic',
		"outpath":'/data/jwp/codes/preTestGen23/argp/argp/third_party/preprocessing/concolic/show_path',
		"Times":3
	}
```

其中:

'concolic_dataset'和'concolic_model'选择固定，'norm'有两个选择即'l0','linf'；

‘basepath’为concolic整个文件夹的地址，所有的预设模型文件、生成的测试用例都将保存在该目录下。

‘outpath'为将展示用的demo储存的目录，该目录位置随意，单独区分该目录的原因是平台前端可能无法读取算法包所处目录（即Concolic目录）。

’Times' 为执行生成的调用次数，每一次调用从数据集中随机抽取少量种子组合然后迭代适当轮数。
