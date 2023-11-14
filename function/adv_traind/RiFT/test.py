from .dataloader import *
from .model import create_model
from os.path import join, dirname
from .train_cifar10 import main as train_cifar10_model
from .at import AdversarialTraining
from .train_rift import main as train_rift_model
import json


def write_json(info, file_name=''):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    return


def start_evaluate(ori_train_epoch=50, at_method='pgd', at_eps=4, at_epoch=20, dataset='cifar10', model_name='resnet18',
                   at_train_epoch=20, rift_epoch=10, batch_size=128,
                   evaluate_methods=['fgsm', 'mifgsm', 'pgd'], save_path='./results/', device='cuda'):
    '''
    :param ori_train_epoch: 正常训练模型轮数，整数
    :param at_method: 对抗训练使用的攻击方法，pgd
    :param at_eps: 对抗训练扰动大小：4
    :param at_epoch: 对抗训练轮数 整数
    :param dataset: 支持数据集类型，[cifar10, cifar100]
    :param model_name: 支持模型类型，[resnet18, vgg19, densenet50]
    :param at_train_epoch: 对抗训练单次训练轮数 整数
    :param rift_epoch: RiFT方法训练轮数，整数
    :param batch_size: 数据批大小
    :param evaluate_methods: 评估模型鲁棒性所使用的对抗攻击方法，支持：fgsm、bim、rfgsm、ffgsm、mifgsm、difgsm、cw、upgd、pgd、tpgd、deepfool等
    :param save_path: 文件保存路径
    :param device: 设备运行类型，cuda或cpu
    :return: 如下所示
            {
                "ori": {
                    "train": 53.576,
                    "test": 54.33,
                    "robust": {
                        "fgsm": 18.92,
                        "mifgsm": 9.6
                    }
                },
                "rift": {
                    "test_acc": 62.02,
                    "robust_acc": {
                        "fgsm": 35.03,
                        "mifgsm": 18.099999999999998
                    }
                }
            }
    '''
    results = dict()
    assert model_name in ['resnet18', 'vgg19', 'densenet50']
    model_names_dict = {'resnet18': 'ResNet18',
                        'vgg19': 'VGG19',
                        'densenet50': 'DenseNet50'}
    model_name = model_names_dict[model_name]
    trainloader, testloader = create_dataloader(batch_size, data_path="./dataset/"+dataset.upper(),dataset=dataset)
    num_classes = 10 if dataset == 'cifar10' else 100
    ori_model = create_model(model_name=model_name, num_classes=num_classes, device='cuda', resume=None)
    ori_results = train_cifar10_model(model=ori_model, trainloader=trainloader,
                                      testloader=testloader, model_save_dir=save_path,
                                      epochs=ori_train_epoch,
                                      device=device,
                                      model_name=f'{dataset}_{model_name}.pth',
                                      attack_methods=evaluate_methods)
    results['ori'] = ori_results['evaluate']
    at = AdversarialTraining(model=ori_results['model'], dataloader=testloader, method=at_method,
                             eps=at_eps, device=device, batch_size=batch_size, train_epoch=at_train_epoch,
                             at_epoch=at_epoch, save_path=save_path, save_name=f'at_{dataset}_{model_name}.pth')
    at_model = at.train()
    rift_layer_dict = {'ResNet18': 'layer2.1.conv2',
                        'VGG19': 'features.46',
                        'DenseNet50': 'dense3.2.conv2'}
    rift_results = train_rift_model(model=at_model,
                                    trainloader=trainloader,
                                    testloader=testloader,
                                    layer=rift_layer_dict[model_name],
                                    epochs=rift_epoch,
                                    model_save_dir=save_path,
                                    attack_method=evaluate_methods,
                                    model_name=model_name,
                                    dataset=dataset,
                                    device=device)
    results['rift'] = rift_results

    print(results)
    write_json(results, join(save_path, f'{dataset}_{model_name}_rift_results.json'))
    return results


if __name__ == '__main__':
    save_path = join(dirname(__file__), 'results')
    dataset, model_name = 'cifar10', 'resnet18'
    start_evaluate(save_path=save_path, dataset=dataset, model_name=model_name)

    dataset, model_name = 'cifar10', 'vgg19'
    start_evaluate(save_path=save_path, dataset=dataset, model_name=model_name)

    dataset, model_name = 'cifar10', 'densenet50'
    start_evaluate(save_path=save_path, dataset=dataset, model_name=model_name)

    dataset, model_name = 'cifar100', 'resnet18'
    start_evaluate(save_path=save_path, dataset=dataset, model_name=model_name)

    dataset, model_name = 'cifar100', 'vgg19'
    start_evaluate(save_path=save_path, dataset=dataset, model_name=model_name)

    dataset, model_name = 'cifar100', 'densenet50'
    start_evaluate(save_path=save_path, dataset=dataset, model_name=model_name)