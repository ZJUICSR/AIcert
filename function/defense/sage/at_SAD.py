from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
AT with sum of absolute values with power p
code from: https://github.com/AberHu/Knowledge-Distillation-Zoo
'''
class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		tmp = fm_s.unsqueeze(1)
		target_size = tmp.size()[2:6]
		loss = F.mse_loss(self.attention_map(fm_s, target_size), self.attention_map(fm_t, target_size))

		return loss

	def attention_map(self, fm, target_size, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p) #指数
		am = torch.sum(am, dim=1, keepdim=True)#维数1的地方求和，如果是二维就是在列上
		am = am.unsqueeze(1)
		Bil = nn.Upsample(size=target_size, mode='trilinear')
		am = Bil(am)
		am = torch.squeeze(am, 1)
		norm = torch.norm(am, dim=(2,3), keepdim=True)#范数
		am = torch.div(am, norm+eps)#除法

		return am