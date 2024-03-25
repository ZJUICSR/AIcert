import sys
import os 


from torch.utils.data import TensorDataset, DataLoader
from function.formal_verify.auto_LiRPA_verifiy.vision.verify import verify as vision_verify
from function.formal_verify.auto_LiRPA_verifiy.language.verify import verify as language_verify
from function.formal_verify.auto_LiRPA_verifiy.language.data_utils import get_sst_data
from function.formal_verify.auto_LiRPA_verifiy.vision.cnn import mnist_model
from function.formal_verify.auto_LiRPA_verifiy.vision.data import get_mnist_data, get_cifar_data, get_gtsrb_data, get_MTFL_data
from function.formal_verify.auto_LiRPA_verifiy.vision.model import get_mnist_cnn_model, get_cifar_resnet18, get_gtsrb_resnet18, get_MTFL_resnet18, get_cifar_densenet_model
from function.formal_verify.auto_LiRPA_verifiy.language.Transformer.Transformer import get_transformer_model
from function.formal_verify.auto_LiRPA_verifiy.language.lstm import get_lstm_demo_model
from function.formal_verify.auto_verify import auto_verify_img
from function.formal_verify.knowledge_consistency import load_checkpoint,get_feature
from function.formal_verify.veritex import Net as reachNet
from function.formal_verify.veritex.networks.cnn import Method as ReachMethod
from function.formal_verify.veritex.utils.plot_poly import plot_polytope2d
