import os
import paddle
print(paddle.__version__)
import paddle.vision.transforms as T
from paddle.vision.transforms import Compose, Normalize
import paddle
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.static import InputSpec
paddle_model_dir='../result_m_v-1/paddle_model'# TODO: you can replace the paddle model path here
import sys
sys.path.append(paddle_model_dir)
from x2paddle_code import ONNXModel
import paddle.nn.functional as F
import numpy as np
batch_size=128

train_dataset = MNIST(mode='train', transform=T.ToTensor())
test_dataset = MNIST(mode='test', transform=T.ToTensor())
paddle.disable_static()
params = paddle.load(os.path.join(os.path.abspath(paddle_model_dir),'model.pdparams'))#TODO: you can replace the paddle model path here
model = ONNXModel()
model.set_dict(params, use_structured_name=True)
model.eval()
accuracies = []
losses = []
test_loader = paddle.io.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)
for batch_id, data in enumerate(test_loader()):
    inputs, labels = data

    predicts = model(inputs.reshape([inputs.shape[0],28,28]))

    loss = F.nll_loss(predicts, labels)
    acc = paddle.metric.accuracy(predicts, labels)    
    
    losses.append(loss.numpy())
    accuracies.append(acc.numpy())

avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
print("validation: loss is: {}, accuracy is: {}".format(avg_loss, avg_acc))