import os
import numpy as np
import pandas as pd
import networkx as nx
import random

from tool import add
# 对每个domain中的Graph，先编码user，再编码item
def load(uId_file, iId_file, warm_user_ids_file, cold_user_ids_file, user_item_npy, user_item_pair_npy, domain='source'):
	G = nx.Graph()
	data_users = pd.read_csv(uId_file, usecols=['ori_uid', 'mapped_uid'])
	data_items = pd.read_csv(iId_file, usecols=['ori_iid', 'mapped_iid'])
	user = dict(zip(data_users.loc[:,'ori_uid'].values.tolist(), data_users.loc[:,'mapped_uid'].values.tolist()))
	item = dict(zip(data_items.loc[:,'ori_iid'].values.tolist(), data_items.loc[:,'mapped_iid'].values.tolist()))

	num_user = data_users.shape[0]
	num_item = data_items.shape[0]
	print('num_user: ',num_user, ' num_item: ',num_item)

	data_cold = pd.read_csv(cold_user_ids_file, usecols=['uid'])
	cold_user_ids = data_cold.loc[:,'uid'].values.tolist()
	num_cold = len(cold_user_ids)
	print('num of cold-start users: ',num_cold)
	data_warm = pd.read_csv(warm_user_ids_file, usecols=['uid'])
	warm_user_ids = data_warm.loc[:,'uid'].values.tolist()
	num_warm = len(warm_user_ids)
	#再加入非shared users
	unshared_uids = list(set(list(user.values())) - set(warm_user_ids) - set(cold_user_ids))
	unshared_uids.sort()
	print('num of unshared users: ',len(unshared_uids))
	# 再加入items
	user_item_pair_dict = np.load(user_item_pair_npy, allow_pickle=True).item()
	user_item_dict = np.load(user_item_npy, allow_pickle=True).item()
	
	sort_item = sorted(list(item.values()))
	re_sort_item = []
	train_user_item_dict = {} #用于train_shared 函数中
	test_user_item_dict = {} #
	train_cross_dict = {}
	val_cross_dict = {}
	train_unshared_data = None # 用于train_unshared函数中

	if domain=='source':
		G.add_nodes_from(warm_user_ids) # 先加入 warm users
		G.add_nodes_from(cold_user_ids) # 再加入 cold users
		G.add_nodes_from(unshared_uids)
		re_sort_item = list(map(lambda x, y : x+y, sort_item, [num_user]*num_item)) #在加入图的时候这个只是节点在图中的一个名字，并不是实际邻接矩阵中的索引
		G.add_nodes_from(re_sort_item)
		# 加边，得到 用于训练的shared user
		for uid in warm_user_ids:
			pos_items = user_item_dict[uid]
			
			negs = list(set(sort_item) - set(pos_items))
			# random.shuffle(negs)
			neg_items = random.sample(negs, len(pos_items))
			neg_items = np.array(neg_items) + num_user
			v = user_item_pair_dict[uid]
			random.shuffle(v)
			v = np.array(v)	
			v[:,1] = v[:,1] + num_user
			G.add_edges_from(v) #加边
			# print(neg_items.shape, '  ',v.shape )
			value = np.concatenate((v,neg_items.reshape(-1,1)),1)
			train_user_item_dict[uid] = value

		# 得到用于训练的single domain的数据
		for uid in cold_user_ids:
			pos_items = user_item_dict[uid]
			negs = list(set(sort_item) - set(pos_items))
			neg_items = random.sample(negs, len(pos_items))
			neg_items = np.array(neg_items) + num_user
			v = user_item_pair_dict[uid]
			random.shuffle(v)
			v = np.array(v)			
			v[:,1] = v[:,1] + num_user
			G.add_edges_from(v)
			# print(neg_items.shape, '  ',v.shape )
			value = np.concatenate((v,neg_items.reshape(-1,1)),1)
			try:
				train_unshared_data = np.concatenate((train_unshared_data,value),0)
			except:
				train_unshared_data = value
		for uid in unshared_uids:
			pos_items = user_item_dict[uid]
			negs = list(set(sort_item) - set(pos_items))
			neg_items = random.sample(negs, len(pos_items))
			neg_items = np.array(neg_items) + num_user
			v = user_item_pair_dict[uid]
			random.shuffle(v)
			v = np.array(v)			
			v[:,1] = v[:,1] + num_user
			G.add_edges_from(v)
			# print(neg_items.shape, '  ',v.shape )
			value = np.concatenate((v,neg_items.reshape(-1,1)),1)
			try:
				train_unshared_data = np.concatenate((train_unshared_data,value),0)
			except:
				train_unshared_data = value			

	elif domain == 'target':
		# target domain在获得embedding matrix的时候，对unshared user和item的embedding索引要减去cold-start users个数
		G.add_nodes_from(warm_user_ids) # 先加入 warm users
		# 不加入 cold users
		G.add_nodes_from(unshared_uids)
		re_sort_item = list(map(lambda x, y : x+y, sort_item, [num_user]*num_item))
		G.add_nodes_from(re_sort_item)
		print('G.number_of_nodes:',G.number_of_nodes())
		# 加边
		for uid in warm_user_ids:
			pos_items = user_item_dict[uid]
			negs = list(set(sort_item) - set(pos_items))
			neg_items = random.sample(negs, len(pos_items))
			neg_items = np.array(neg_items) + num_user - num_cold
			v = user_item_pair_dict[uid]
			v = np.array(v)			
			v[:,1] = v[:,1] + num_user
			G.add_edges_from(v) #加边
			v[:,1] = v[:,1] - num_cold
			value = np.concatenate((v, neg_items.reshape(-1,1)),1)

			train_user_item_dict[uid] = value
			num_train_cross = int(len(pos_items) * 0.9)
			train_cross_dict[uid] = value[:num_train_cross,:]
			if (len(pos_items) - num_train_cross) > 0:
				pos_val = pos_items[num_train_cross:]
				pos_val = np.array(pos_val) + num_user - num_cold
				neg_val = random.sample(negs, 99)
				neg_val = np.array(neg_val) + num_user - num_cold
				val_cross_dict[uid] = [pos_val, neg_val]
		for uid in cold_user_ids:
			pos_test = user_item_dict[uid]
			random.shuffle(pos_test)
			negs = list(set(sort_item) - set(pos_test))
			random.shuffle(negs)
			neg_items = random.sample(negs, 999)
			neg_test = np.array(neg_items) + num_user - num_cold
			pos_test = np.array(pos_test) + num_user - num_cold

			test_user_item_dict[uid] = [pos_test, neg_test]

		for uid in unshared_uids:
			pos_items = user_item_dict[uid]
			negs = list(set(sort_item) - set(pos_items))
			neg_items = random.sample(negs, len(pos_items))
			neg_items = np.array(neg_items) + num_user - num_cold
			v = user_item_pair_dict[uid]
			random.shuffle(v)
			v = np.array(v)			
			v[:,1] = v[:,1] + num_user
			G.add_edges_from(v)
			v[:,0] = v[:,0] - num_cold
			v[:,1] = v[:,1] - num_cold
			value = np.concatenate((v,neg_items.reshape(-1,1)),1)
			try:
				train_unshared_data = np.concatenate((train_unshared_data,value),0)
			except:
				train_unshared_data = value

	else: 
		print('Domain Type Error')
		exit()
	print('num of nodes in G: ', G.number_of_nodes())
	print('num of edges in G: ', G.number_of_edges())
	if domain == 'source':
		return num_user, num_item, G, warm_user_ids, cold_user_ids, unshared_uids, train_user_item_dict, train_unshared_data
	elif domain == 'target':
		return num_user, num_item, G, warm_user_ids, cold_user_ids, unshared_uids, train_user_item_dict, test_user_item_dict, train_unshared_data, train_cross_dict, val_cross_dict
	
	


