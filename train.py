import os
import numpy as np
import itertools
import argparse
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch.autograd import Variable

from encoder import Encoder
from decoder import Decoder
from tool import bpr_loss, mse_loss, BCELoss,l1_loss

def train_unshared(args, encoder,optim_pre, userIdx, posIdx, negIdx,reg=1e-4):
   	_,emb = encoder(only_common=True)
	user_emb = emb[userIdx]
	pos_item_emb = emb[posIdx]
	neg_item_emb = emb[negIdx]
	loss, reg_loss, pre_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb,reg=reg)
	optim_pre.zero_grad()
	loss.backward()
	optim_pre.step()
	if hasattr(torch.cuda, 'empty_cache'):
		torch.cuda.empty_cache()
	return loss.item(), reg_loss.item(),pre_loss.item()

def train_shared(args,optims, inputsX, inputsY, encoderX2Y, encoderY2X, decoderX2Y, decoderY2X, shared_users, train_user_item_dictX, train_user_item_dictY, train_cross_dict,reg=5e-4):
	torch.autograd.set_detect_anomaly(True)
	optim_pre_X, optim_pre_Y, optim_pre_X2Y, optim_pre_Y2X,optim_auto_X,optim_auto_Y,optim_superX2Y,optim_superY2X = optims

	# 这个target是decoder输出的重建初始特征矩阵的
	targetsX = inputsY # 把shared user编号在最前边
	targetsY = inputsX
	
	# encoder
	RX, common_embX = encoderX2Y(shared_users)
	RY, common_embY = encoderY2X(shared_users)
	
	# decoder
	outputsX2Y_1, outputsX2Y_2  = decoderX2Y(RX) #这里decoder重建的是GCN的输出，相当于是节点的latent factor
	outputsY2X_1, outputsY2X_2 = decoderY2X(RY)
	
	#autoencoders
	_, auto_outputX = decoderY2X(RX)
	_, auto_outputY = decoderX2Y(RY)
	##### Losses
	#Loss of cross-domain supervised loss #
	loss_superX2Y = mse_loss(outputsX2Y_2, targetsX) * args.lambda_l1
	loss_superY2X = mse_loss(outputsY2X_2, targetsY) * args.lambda_l1 

	# autoencoder loss
	loss_autoX = mse_loss(auto_outputX, inputsX) * args.lambda_l1
	loss_autoY = mse_loss(auto_outputY, inputsY) * args.lambda_l1
	
	# link prediction loss
	tripleX = None # (N,3)
	tripleY = None
	tripleX2Y = None
	user_embX2Y = None
	user_embY2X = None

	for i,u in enumerate(shared_users): #Is there any other better method instead of traversal
		u = int(u.detach().cpu().numpy())
		udataX = torch.LongTensor(train_user_item_dictX[u]).cuda()
		udataY = torch.LongTensor(train_user_item_dictY[u]).cuda()
		udataX2Y = torch.LongTensor(train_cross_dict[u]).cuda()

		try:
			tripleX = torch.cat((tripleX,udataX),0)
			tripleY = torch.cat((tripleY,udataY),0)
			tripleX2Y = torch.cat((tripleX2Y, udataX2Y), 0)
			user_embX2Y = torch.cat((user_embX2Y,outputsX2Y_1[i].expand(len(udataX2Y),outputsX2Y_1[i].shape[0])),0)
			user_embY2X = torch.cat((user_embY2X,outputsY2X_1[i].expand(len(udataX),outputsY2X_1[i].shape[0])),0)

		except:
			tripleX = udataX
			tripleY = udataY
			tripleX2Y = udataX2Y

			user_embX2Y = outputsX2Y_1[i].expand(len(udataX2Y),outputsX2Y_1[i].shape[0])
			user_embY2X = outputsY2X_1[i].expand(len(udataX),outputsY2X_1[i].shape[0])
	
	assert len(user_embY2X) == len(tripleX), 'len(user_embY2X) != len(tripleX)'		
	
	# Loss of link prediction in single domain, bpr loss #
	user_embX = common_embX[tripleX[:,0]]
	pos_item_embX = common_embX[tripleX[:,1]]
	neg_item_embX = common_embX[tripleX[:,2]]
	loss_preX, loss_regX, _ = bpr_loss(user_embX, pos_item_embX, neg_item_embX,reg=reg) 
	
	user_embY = common_embY[tripleY[:,0]]
	pos_item_embY = common_embY[tripleY[:,1]]
	neg_item_embY = common_embY[tripleY[:,2]]	
	loss_preY, loss_regY, _ = bpr_loss(user_embY, pos_item_embY, neg_item_embY,reg=reg)

	# Loss of link prediction for cross domain, bpr loss
	pos_item_embX2Y = common_embY[tripleX2Y[:,1]]
	neg_item_embX2Y = common_embY[tripleX2Y[:,2]]
	loss_preX2Y, loss_regX2Y, _ = bpr_loss(user_embX2Y, pos_item_embX2Y, neg_item_embX2Y,reg=reg)
	loss_preY2X, loss_regY2X, _ = bpr_loss(user_embY2X, pos_item_embX, neg_item_embX,reg=reg)

	##### Optimizator
	optim_pre_X.zero_grad()
	loss_preX.backward(retain_graph=True)
	optim_pre_X.step()
	
	optim_pre_Y.zero_grad()
	loss_preY.backward(retain_graph=True)
	optim_pre_Y.step()

	optim_pre_X2Y.zero_grad()
	loss_preX2Y.backward(retain_graph=True)
	optim_pre_X2Y.step()

	optim_pre_Y2X.zero_grad()
	loss_preY2X.backward(retain_graph=True)
	optim_pre_Y2X.step()


	optim_auto_X.zero_grad()
	loss_autoX.backward(retain_graph=True) #retain_graph=True
	optim_auto_X.step()

	optim_auto_Y.zero_grad()
	loss_autoY.backward(retain_graph=True)
	optim_auto_Y.step()

	optim_superX2Y.zero_grad()
	loss_superX2Y.backward(retain_graph=True)
	optim_superX2Y.step()	

	optim_superY2X.zero_grad()
	loss_superY2X.backward()
	optim_superY2X.step()

	del outputsX2Y_2
	del outputsY2X_2

	loss_all = (loss_preX + loss_preY) + (loss_preX2Y + loss_preY2X) + (loss_superX2Y + loss_superY2X) + (loss_autoX + loss_autoY)
	if hasattr(torch.cuda, 'empty_cache'):
		torch.cuda.empty_cache()
	return loss_all.item(), loss_preX.item(),loss_regX.item(),loss_preY.item(),loss_regY.item(), loss_preX2Y.item(), loss_preY2X.item(),loss_regX2Y.item(),loss_regY2X.item(), 0,0, loss_superX2Y.item(), loss_superY2X.item()
