import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch.autograd import Variable
import dgl
from dgl.nn.pytorch import GraphConv
from pytorch_revgrad import RevGrad

class Decoder(nn.Module):
	def __init__(self, in_dim, out_dim1, out_dim2):
		super(Decoder,self).__init__()
		self.in_dim = in_dim
		self.out_dim1 = out_dim1 # 重建GCN的维度
		self.out_dim2 = out_dim2 # 重建原始特征的维度
		self.fc3 = nn.Linear(out_dim1, out_dim2) #重建原始的特征
		self.tanh = nn.Tanh()
		self.layers = nn.Sequential(
							nn.Linear(in_dim, 64),
							nn.Tanh(),
							nn.Linear(64, out_dim1)
							)

	def forward(self, x):
		recon_gcn = self.layers(x) #重建GCN的输出		
		hidden2 = self.tanh(recon_gcn)
		output = self.fc3(hidden2)
		return recon_gcn, output
