# -*- coding: utf-8 -*-
import torch

from fvcore.nn import FlopCountAnalysis, parameter_count_table
from paca_detect.net import TwoStraeamSrncovetTest
from paca_detect.resnet_cbam import TwoStreamCBAMTest
from thop import profile
import time
import numpy as np


def fps(model, img, device='cuda'):
    torch.cuda.synchronize()
    start = time.time()
    _ = model(img.to(device))
    torch.cuda.synchronize()
    end = time.time()
    return end - start


def test(device='cuda'):
    # 创建输入网络的tensor
    tensor = torch.rand(1, 3, 224, 224).to(device)

    cbam = TwoStreamCBAMTest().to(device)
    paca = TwoStraeamSrncovetTest().to(device)
    # 分析FLOPs
    # print(f'cbam={cbam(tensor)}, paca={paca(tensor)}')
    flops = FlopCountAnalysis(cbam, tensor)
    print("CBAM FLOPs: ", flops.total())

    flops = FlopCountAnalysis(paca, tensor)
    print("paca FLOPs: ", flops.total())

    # 分析parameters
    # print(parameter_count_table(cbam))
    # print(parameter_count_table(paca))

    macs, params = profile(cbam, inputs=(tensor,))
    print(f'cbam macs={macs}, params={params}')
    macs, params = profile(paca, inputs=(tensor,))
    print(f'paca macs={macs}, params={params}')

    # cbam_fps = [fps(cbam, tensor) for _ in range(1000)]
    # paca_fps = [fps(paca, tensor) for _ in range(1000)]
    # print(f'cbam_fps={np.array(cbam_fps).min()} - {np.array(cbam_fps).max()}')
    # print(f'paca_fps={np.array(paca_fps).min()} - {np.array(paca_fps).max()}')




if __name__ == '__main__':
    test()

