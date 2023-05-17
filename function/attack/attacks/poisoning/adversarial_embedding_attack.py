from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Tuple, Union, List, Optional, TYPE_CHECKING
import numpy as np
import torch
import torch.optim as optim
from function.attack.attacks.attack import PoisoningAttackTransformer
from function.attack.attacks.poisoning import PoisoningAttackBackdoor
from function.attack.estimators.classification.pytorch import PyTorchClassifier
from function.attack.attacks.utils import compute_success, compute_accuracy, IntermediateLayerGetter
# KerasClassifier
# from function.attack.estimators.classification.keras import KerasClassifier
if TYPE_CHECKING:
    from function.attack.attacks.utils import CLASSIFIER_NEURALNETWORK_TYPE, CLASSIFIER_TYPE
logger = logging.getLogger(__name__)

class NewModel(torch.nn.Module):
    def __init__(self, net, input_test, device):
        super(NewModel, self).__init__()
        self.device = device
        self.net = net
        # 使得self.net.feature被初始化
        # 构建的网络在进行鲁棒测试时必须要在forward中写入self.feature作为特征层的输出
        # self.feature需要保持的shape为(batch_size, 特征数目)，即特征应当是被平铺操作后的值
        # 参见resnet中的实现
        test = self.net.forward(torch.from_numpy(input_test).to(self.device))
        self.in_features = self.net.feature.shape[1]

        self.relu = torch.nn.LeakyReLU()
        self.fc1 = torch.nn.Linear(self.in_features, 256, bias=True)
        self.bn1 = torch.nn.BatchNorm1d(num_features=256)
        self.fc2 = torch.nn.Linear(256, 128, bias=True)
        self.bn2 = torch.nn.BatchNorm1d(num_features=128)
        self.fc3 = torch.nn.Linear(128, 2, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, epoch_num:int):
        # self.net.train(mode=True)
        # self.train(mode=True)
        y1 = self.net.forward(x)
        x1 = self.fc1(self.net.feature + ((0.1/epoch_num)**0.5)*torch.randn_like(self.net.feature))
        x1 = self.relu(x1)
        self.bn1(x1)
        x1 = self.fc2(x1)
        x1 = self.relu(x1)
        self.bn2(x1)
        x1 = self.fc3(x1)
        y2 = self.softmax(x1)
        return y1, y2
        

class PoisoningAttackAdversarialEmbedding(PoisoningAttackTransformer):
    attack_params = PoisoningAttackTransformer.attack_params + [
        "backdoor",
    ]

    _estimator_requirements = (PyTorchClassifier, )

    def __init__(
        self,
        backdoor: PoisoningAttackBackdoor,
        device: torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        # super().__init__(classifier=classifier)
        self.device = device
        self.backdoor = backdoor
        self.device = device
        
    
    def poison_estimator(self, x: np.ndarray, y: np.ndarray, **kwargs) -> "CLASSIFIER_TYPE":
        return super().poison_estimator(x, y, **kwargs)
    
    # 训练数据投毒
    def poison (  # pylint: disable=W0221
        self, x: np.ndarray, y: Optional[np.ndarray] = None, broadcast=False, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.backdoor.poison(x, y, broadcast=True)
    
    # 重新训练得到被增强后的投毒模型
    def fintune(self, model, x_train: np.ndarray, y_train: np.ndarray, select_list: list, num_epochs: int=20, batch_size: int=700, lr=0.001, alpha=1):
        # architecture for discriminators
        self.model = model

        input_test=x_train[0]
        input_test=input_test[np.newaxis, :]
        input_test[0,:] = 1

        self.new_model = NewModel(self.model, input_test=input_test, device=self.device).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.new_model.parameters(), lr=lr)
        # 干净样本和投毒样本的标签
        B = np.ones((len(x_train)))
        for i in select_list:
            B[i] = 0

        num_batch = int(np.ceil(len(x_train) / float(batch_size)))
        total_step = len(x_train)
        for epoch in range(num_epochs):
            train_loss1 = 0
            train_loss2 = 0
            for m in range(num_batch):
                # Batch indexes
                optimizer.zero_grad()
                begin, end = (
                    m * batch_size,
                    min((m + 1) * batch_size, x_train.shape[0]),
                )
                # 前向传播
                model_outputs1, model_outputs2 = self.new_model.forward(torch.from_numpy(x_train[begin:end]).to(self.device), epoch_num=epoch+1)
                # 损失计算
                loss1 = criterion(model_outputs1, torch.from_numpy(y_train[begin:end]).to(self.device))
                loss2 = criterion(model_outputs2, torch.from_numpy(B[begin:end]).to(self.device).to(torch.int64))
                loss = loss1 - alpha * loss2
                loss.backward()
                optimizer.step()
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()

            self.classifier = PyTorchClassifier (
                model=model,
                clip_values=(0.0, 1.0),
                loss=criterion,
                optimizer=optimizer,
                input_shape=(3, 32, 32),
                nb_classes=10,
                device=self.device
            )
            acc, _ = compute_accuracy(self.classifier.predict(x_train, batch_size=batch_size), y_train)
            print("Epoch [{}/{}], train_loss: {:.6f}, discriminator_loss: {:.16f}, accuracy: {:.6f}".format(epoch+1, num_epochs, train_loss1 / total_step, train_loss2 / total_step, acc))
