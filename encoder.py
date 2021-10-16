import os
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy.spatial.distance import pdist
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch.autograd import Variable
import dgl
from dgl.nn.pytorch import GraphConv
from tool import bpr_loss

class Encoder(nn.Module):
	def __init__(self,
				g,
				features,
				# userNum,
				# itemNum,
				in_dim,
				n_hiddens, # a list
				out_dim,
				n_layers,
				activation,
				dropout):
		super(Encoder,self).__init__()
		self.g = g
		self.features = features	
		self.layers = nn.ModuleList()
		self.layers.append(GraphConv(in_dim, n_hiddens[0], activation=activation))
		for i in range(n_layers - 1):
			self.layers.append(GraphConv(n_hiddens[i], n_hiddens[i+1], activation=activation))
		self.dropout = nn.Dropout(p=dropout)
		self.SR_layer = nn.Sequential(
						nn.Linear(n_hiddens[-1], 64),
						nn.Tanh(),
						nn.Linear(64, out_dim)
						)		

	def forward(self, users=None, mapped_feature=None, isreplace=False, only_common=False):
		# 这里的userIdx和pos_itemIdx(or neg_itemIdx)是成对儿的，
		
		if isreplace:
			h = self.features.clone()
			h[users] = mapped_feature
		else:
			h = self.features.clone()
		for i, layer in enumerate(self.layers):
			if i != 0:
				h = self.dropout(h)
			h = layer(self.g, h)
		user_h = h[users]
		R = self.SR_layer(user_h)
		return R, h
