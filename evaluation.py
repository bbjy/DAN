import os
import numpy as np
import itertools
import argparse
from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch.autograd import Variable
import datetime 

from encoder import Encoder
from decoder import Decoder
from tool import bpr_loss, l1_loss, BCELoss
from metrics import hit_at_k, ndcg_at_k

def evaluate(args, encoderX2Y, encoderY2X, decoder, cold_users, test_user_item_dictX2Y,test=False):
	# domain X (source) -> domain Y (target)
	hit_rate_result = defaultdict(list)
	ndcg_result = defaultdict(list)
	user_embX2Y = None
	with torch.no_grad():
		# encoder
		encoderX2Y.eval()
		encoderY2X.eval()
		decoder.eval()
		RX, common_embX = encoderX2Y(cold_users)
		_,common_embY = encoderY2X(only_common=True)
		# decoder
		outputsX2Y,_ = decoder(RX)
		if test:
			np.save('./embs/outputsX2Y.npy',outputsX2Y.detach().cpu().numpy())
			np.save('./embs/common_embY.npy',common_embY.detach().cpu().numpy())
	
		for i,u in enumerate(cold_users):
			u = int(u.detach().cpu().numpy())
			pos_item_ids = torch.LongTensor(test_user_item_dictX2Y[u][0]).cuda()
			neg_item_ids = torch.LongTensor(test_user_item_dictX2Y[u][1]).cuda()
	
			pos_item_embY = common_embY[pos_item_ids]
			neg_item_embY = common_embY[neg_item_ids]
			test_item_embY = torch.cat((pos_item_embY, neg_item_embY),0)
			user_embX2Y = outputsX2Y[i].repeat(len(test_item_embY),1)
	
			test_rating = torch.sigmoid(torch.sum(torch.mul(user_embX2Y, test_item_embY), axis=1))
			test_pred = torch.squeeze(test_rating)
			for k in args.topk:
				_,pred_top_k_position = torch.topk(test_pred, k, largest=True, sorted=True) # 返回前k个最大值排序后的下标
				pred_top_k_position = pred_top_k_position.tolist()
				truth_position = list(range(len(pos_item_embY)))
				hit_rate_result[k].append(hit_at_k(pred_top_k_position, truth_position, k))
				ndcg_result[k].append(ndcg_at_k(pred_top_k_position, truth_position, k))
		
		hit_rate = []
		ndcg = []
		for k in args.topk:
			hit_rate.append(np.mean(hit_rate_result[k]))
			ndcg.append(np.mean(ndcg_result[k]))
		return hit_rate, ndcg


