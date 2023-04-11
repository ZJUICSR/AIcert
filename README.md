# 对抗样本攻击&&后门攻击

## 攻击列表
白盒攻击                            初始接口
FGSM                                OK
BIM                                 OK
PGD                                 OK
C&W                                 OK
DeepFool                            OK
JacobianSaliencyMap                 OK
UniversalPerturbation               OK
AutoAttack                          OK 
GD-UAP                              OK

黑盒攻击
SquareAttack                        OK
HSJA                                OK      
PixelAttack                         OK
SimBA                               OK
BoundaryAttack                      OK
ZOO                                 OK
GeoDA                               OK  
Fastdrop                            OK

投毒
BackdoorAttack                      OK
Clean-LabelBackdoorAttack           OK
CleanLabelFeatureCollisionAttack    OK
AdversarialBackdoorEmbedding        OK

## 数据集
（mnist cifar10）文件夹datasets中，可以自动完成下载

## 模型
（resnet，lenet）文件夹models中，model_net子文件夹存放模型代码，model_ckpt存放训练得到的模型参数
