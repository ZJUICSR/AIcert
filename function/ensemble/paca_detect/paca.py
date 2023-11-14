
import torch
import torch.optim as optim
import torch.nn as nn
from .data import get_adv_dataloader, combine_ori_adv_dataloader
from .net import TwoStraeamSrncovet
import os
import json
import copy
from torch.utils.data import DataLoader
from .resnet_model import get_resnet50_model
from tqdm import tqdm
import warnings


warnings.filterwarnings('ignore')


MODEL_SAVE_PATH = os.path.join('output/cache', 'detect_models')
RESULT_SAVE_PATH = os.path.join('output/cache', 'results')

for path in [MODEL_SAVE_PATH, RESULT_SAVE_PATH]:
    if not os.path.exists(path):
        os.mkdir(path)


def train_single_epoch(detect_model,
                       target_model,
                       dataloader: DataLoader,
                       lr_scheduler,
                       loss_func,
                       optimizer,
                       epoch: int,
                       device='cuda'):
    detect_model.train()
    lr_scheduler.step(epoch=epoch)
    criterion = loss_func
    train_loss = 0
    train_acc = 0
    # for batch_idx, data in tqdm(enumerate(dataloader), ncols=100, desc=f'Train epoch {epoch}'):
    for batch_idx, data in enumerate(dataloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        images = torch.autograd.Variable(images, requires_grad=True)
        y_logit = target_model(images)
        _, pred_class = torch.max(y_logit, 1)
        loss = criterion(y_logit, pred_class)
        gradient = torch.autograd.grad(loss, images)[0]
        gradient = torch.abs(gradient).detach().to(device)
        # fuse_input = torch.cat((images,gradient),1)

        mini_out = detect_model(images, gradient)
        mini_loss = criterion(mini_out, labels.long())
        mini_loss.backward()
        optimizer.step() 
        
        _, pred = torch.max(mini_out, 1)
        acc = (pred.data == labels.long()).sum()
        train_loss += float(mini_loss)
        train_acc += float(acc)
        # if batch_idx % 500 == 0:
        #     print('datashapes:', images.shape, labels.shape)
        #     print('labels:', labels)
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(labels)}/{len(dataloader.sampler)} '
        #           f'({00. * batch_idx / len(dataloader):.0f}%)]\t'
        #           f'Loss: {mini_loss / len(labels):.6f}  Acc:{float(acc) / len(labels):.6f}')

    # print(f'====> Epoch: {epoch} Average loss: {train_loss / (len(dataloader.sampler)):.6f}'
    #       f' Average accuracy: {train_acc / (len(dataloader.sampler)):.6f}')
    # torch.save(model.state_dict(), 'pretrained/model'+epoch+'.pkl')

    torch.cuda.empty_cache()
    return round(train_acc / len(dataloader.sampler), 6)


def train_adv_detect_model(attack,
                           target_model,
                           dataloader: DataLoader,
                           max_epochs=30,
                           target_acc=0.90,
                           device='cuda',
                           log_func=None,
                           channel=3):
    torch.manual_seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    detect_model = TwoStraeamSrncovet(in_channel=channel)
    detect_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(detect_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 70, 150], gamma=0.1)

    ori_acc, epoch_acc = 0, 0
    for epoch_num in range(max_epochs):
        epoch_acc = train_single_epoch(detect_model=detect_model,
                                       target_model=target_model,
                                       dataloader=dataloader,
                                       lr_scheduler=lr_scheduler,
                                       loss_func=criterion,
                                       optimizer=optimizer,
                                       epoch=epoch_num,
                                       device=device)
        if epoch_acc > ori_acc:
            ori_acc = epoch_acc
            torch.save(detect_model.state_dict(), os.path.join(MODEL_SAVE_PATH, f'{attack}_detect.pkl'))
        if log_func is not None:
            log_func(f'[模型测试阶段] {attack}攻击监测模型第 {epoch_num + 1} epoch训练结束，当前准确率：{epoch_acc}')
        if epoch_acc >= target_acc:
            break

    return {attack: ori_acc, 'model': detect_model}


def save_json_info(filename, info):
    result = dict()
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            result = json.load(f)
    result['PACA'] = info

    with open(filename, 'w', encoding='utf-8') as file_obj:
        json.dump(result, file_obj, indent=4, ensure_ascii=False)
    return


def json2dict(file_path) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        result = json.load(file)
    return result


def load_cache(file_name, save_file):
    result = json2dict(file_name)
    save_json_info(save_file, result)
    return


def get_upload_acc(ori_model, upload_info: dict, device='cuda', log_func=None, channel=3):
    '''
    :param ori_model: 原始模型
    :param upload_info: 上传的数据集
    :param device: 设备类型
    :param log_func: 日志记录函数
    :return: test_acc：测试集上的分类准确率
    '''
    attack_model = copy.deepcopy(ori_model)
    attack_model.eval()
    attack_model.to(device)
    attack_model.retain_graph = True

    train_dataloader = combine_ori_adv_dataloader(ori_dataloader=upload_info['train'][0],
                                                  adv_dataloader=upload_info['train'][1])
    test_dataloader = combine_ori_adv_dataloader(ori_dataloader=upload_info['test'][0],
                                                 adv_dataloader=upload_info['test'][1])
    method = 'UPLOAD'
    if log_func is not None:
        log_func(f"[模型测试阶段] 开始训练上传数据PACA监测模型")
    attack_result = train_adv_detect_model(attack=method,
                                           target_model=attack_model,
                                           dataloader=train_dataloader,
                                           device=device,
                                           log_func=log_func,
                                           channel=channel)
    attack_acc = attack_result[method]
    detect_model = attack_result['model']
    if log_func is not None:
        log_func(f"[模型测试阶段] 上传信息监测模型训练结束，准确率为：{attack_acc}，开始在测试集上验证模型准确率")

    detect_model.eval()
    criterion = nn.CrossEntropyLoss()
    test_acc = 0
    for images, labels in tqdm(test_dataloader, ncols=100, desc='测试PACA模型上传数据监测准确率'):
        images, labels = images.to(device), labels.to(device)
        images = torch.autograd.Variable(images, requires_grad=True)
        y_logit = attack_model(images)
        _, pred_class = torch.max(y_logit, 1)
        loss = criterion(y_logit, pred_class)
        gradient = torch.autograd.grad(loss, images)[0]
        gradient = torch.abs(gradient).detach().to(device)
        # fuse_input = torch.cat((images,gradient),1)

        mini_out = detect_model(images, gradient)
        _, pred = torch.max(mini_out, 1)
        acc = (pred.data == labels.long()).sum()

        test_acc += float(acc)
    test_acc = round(test_acc / len(test_dataloader.sampler), 6)
    if log_func is not None:
        log_func(f"[模型测试阶段] PACA监测模型在测试集上的准确率为：{test_acc}")

    return test_acc


def paca_detect(ori_model,
                ori_dataloader,
                attack_info: list,
                upload_info: dict,
                param_hash=None,
                device='cuda',
                save_path=RESULT_SAVE_PATH,
                log_func=None,
                channel=3
                ):
    '''
    :param ori_model: 原始模型
    :param ori_dataloader: 原始数据集
    :param attack_info: [{"method": attack1, "dataloader": dataloader1},
                         {"method": attack2, "dataloader": dataloader2},
                         ...]， 其中，method的值为攻击类型（FGSM、DeepFool等），
                         dalaloader的值为原始数据集在该攻击下产生的dataloader
    :param upload_info: 自定义上传数据，
    数据内容{"train": [ben_dataloader, adv_dataloader], "test":[ben_dataloader, adv_dataloader]}
    :param param_hash: 参数hash值，读取缓存时使用
    :param device: 设备类型，默认使用GPU
    :param save_path: 结果保存到save_path下的“result.json”文件
    :param log_func:  日志记录函数，默认为空，函数输入类型为字符串
    :return: None
    '''

    cache_file = 'cache.json'
    result_file = 'keti4.json'
    cache_path = os.path.join(RESULT_SAVE_PATH, f'{param_hash}')
    # print(f'cache_path={cache_path}')
    if param_hash is not None \
            and os.path.exists(os.path.join(cache_path, cache_file)):  # 直接读取缓存结果
        # print(f'param_hash={param_hash}, 读取缓存结果')
        load_cache(os.path.join(cache_path, cache_file),
                   os.path.join(save_path, result_file))
        return
    if log_func is not None:
        log_func('[模型测试阶段] 启动课题四自动化攻击监测PACA模型训练')
    attack_model = copy.deepcopy(ori_model)
    attack_model.eval()
    attack_model.to(device)
    attack_model.retain_graph = True

    result = dict()
    for attacks in attack_info:
        if log_func is not None:
            log_func(f"[模型测试阶段] 开始训练{attacks['method']}攻击监测模型")
        test_dataloader = combine_ori_adv_dataloader(ori_dataloader, attacks['dataloader'])
        attack_result = train_adv_detect_model(attack=attacks['method'],
                                               target_model=attack_model,
                                               dataloader=test_dataloader,
                                               device=device,
                                               log_func=log_func,
                                               channel=channel)
        attack_acc = attack_result[attacks['method']]
        if log_func is not None:
            log_func(f"[模型测试阶段] {attacks['method']}监测模型训练结束，准确率为：{attack_acc}")

        result.update({attacks['method']: attack_acc})

    if upload_info is not None:
        upload_acc = get_upload_acc(ori_model=ori_model, upload_info=upload_info, device=device, log_func=log_func)
        result.update({'UPLOAD': upload_acc})

    save_json_info(os.path.join(save_path, result_file), result)
    if log_func is not None:
        log_func(f"[模型测试阶段] 课题四自动化攻击监测PACA模型训练结束。")

    # 保存缓存
    if param_hash is not None:
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        # print(f'param_hash={param_hash}, 保存缓存结果')
        save_json_info(os.path.join(cache_path, cache_file), result)
    print("paca:", result)
    return result

def demo():
    train_loader, ver_loader, _ = get_adv_dataloader(batch_size=1)
    device = 'cpu'
    ori_model = get_resnet50_model()

    paca_detect(ori_model=ori_model,
                ori_dataloader=train_loader,
                attack_info=[{"method": 'FGSM', 'dataloader': ver_loader}],
                param_hash='12345',
                device=device,
                save_path=RESULT_SAVE_PATH,
                log_func=print)


if __name__ == '__main__':
    demo()
