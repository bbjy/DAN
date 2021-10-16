import os
import numpy as np
import torch
import math
def hit_at_k(pred_top_k_ids, test_item_list,k):
	hits = 0
	for i in pred_top_k_ids:
		if i in test_item_list:
			hits += 1
	hit_at_k = hits / min(float(len(test_item_list)),k)
	return hit_at_k

def ndcg_at_k(pred_top_k_ids, test_item_list, k):
	dcgs_at_k = 0
	n = len(test_item_list)
	l = min(k,n)
	for i,j in zip(pred_top_k_ids, range(k)):
		if i in test_item_list:
			dcgs_at_k += (1.0 / math.log(j+2,2))
	idcg = sum(1.0/ math.log(i+2,2) for i in range(l))
	ndcg_at_k = dcgs_at_k / idcg
	return ndcg_at_k