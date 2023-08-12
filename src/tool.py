import os
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse import vstack

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch.autograd import Variable
import dgl

def add(x, y):
	return x+y

def bpr_loss(user_emb, pos_item_emb, neg_item_emb, reg=0.0001):
	pos_ratings = torch.sum(torch.mul(user_emb, pos_item_emb),axis=1)
	neg_ratings = torch.sum(torch.mul(user_emb, neg_item_emb),axis=1)

	pre_loss = F.sigmoid(pos_ratings - neg_ratings).log()
	l2_norm = torch.norm(torch.cat([user_emb, pos_item_emb, neg_item_emb]), 2)
	l2_norm = torch.pow(l2_norm, 2)
	reg_loss = reg * l2_norm
	pre_loss =  - torch.sum(pre_loss)
	loss = reg_loss + pre_loss
	return loss,reg_loss, pre_loss

def l1_loss(inputs, targets):
	# l1loss = nn.L1Loss(reduction='mean')
	l1loss = nn.SmoothL1Loss(reduction='mean')
	loss = l1loss(inputs, targets)
	return loss
def mse_loss(inputs, targets):
	mseloss = nn.MSELoss(reduce=True, size_average=True)
	loss = mseloss(inputs,targets)
	return loss

def BCELoss(pred, truth):
	loss_fun = nn.BCEWithLogitsLoss()
	loss = loss_fun(pred, truth)
	return loss

def early_stopping(cur_ndcg, cur_best_ndcg, stopping_step, flag_step):
	'''
	cur_ndcg: 当前的ndcg值
	cur_best_ndcg: 目前为止最好的ndcg值
	stopping_step: ndcg值没有提高的步数
	flag_step: 如果ndcg连续flag_step步都没有提升，则执行early_stopping
	'''
	if cur_ndcg >= cur_best_ndcg:
		stopping_step = 0
		cur_best_ndcg = cur_ndcg
	else:
		stopping_step += 1
	if stopping_step >= flag_step:
		should_stop = True
	else:
		should_stop = False
	return cur_best_ndcg, stopping_step, should_stop
