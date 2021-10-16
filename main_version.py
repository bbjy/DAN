import os
import numpy as np
import scipy.sparse as sp
import random
from copy import deepcopy
from collections import OrderedDict
from random import shuffle
import torch
import networkx as nx
import dgl
from torch.nn import functional as F
import time
import datetime
import argparse
from collections import defaultdict
import gc
from termcolor import colored, cprint
import itertools

from load_data import load
from encoder import Encoder
from decoder import Decoder
from train import train_unshared, train_shared
from tool import bpr_loss, l1_loss, BCELoss, early_stopping
from evaluation import evaluate
from tensorboardX import SummaryWriter
writer =  SummaryWriter() #使用时间命名保存在runs/文件夹下

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=64, help='batch size of users for cross-domain training')
parser.add_argument('--batch_links', type=int, default=128, help='batch size of links for single-domain link prediction training')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--decay', type=float, default=0.0, help='weight deacy of adam optimizer')
parser.add_argument('--bpr_reg', type=float, default=5e-4, help='weight deacy of adam optimizer')
parser.add_argument('--hidden', nargs='+', type=int, default=[64,64], help='hidden layers')
parser.add_argument('--n_layers', type=int, default=2, help='number of hidden layers')
parser.add_argument('--out_dim_encode', type=int, default=32, help='dimension of output of encoder')
parser.add_argument('--lambda_l1', type=float, default=1.0, help='weight of L1 loss')
parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight of GAN loss')
parser.add_argument('--lambda_hsic', type=float, default=1.0, help='weight of hsic loss')
parser.add_argument('--beta', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--topk', type=list, default=[5,10,20], help='Evaluation topk results.')
parser.add_argument('--activation_gcn', type=str, default='leaky_relu_', help='Activation function of GCN.')
parser.add_argument('--activation_mlp', type=str, default='tanh', help='Activation function of mlp layers.')
parser.add_argument('--user_emb_mode', type=str, default='cml', help='mode of initial user embedding.')
parser.add_argument('--cold_ratio', type=float, default=0.5, help='Cold ratio.')
parser.add_argument('--source', type=str, default='Book', help='source domain name.')
parser.add_argument('--target', type=str, default='CD', help='target domain name.')
parser.add_argument('--test', action='store_true', help='Testing model')
parser.add_argument('--normalize', action='store_true', help='normalize initial embedding')
parser.add_argument('--load_model', action='store_true', help='Load saved model')
parser.add_argument('--device', type=str, default='0', help='use GPU computation')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device


if __name__ == '__main__':
	if args.source == 'Book':
		path = 'path to back2cd'
	else:
		path = 'path to cd2movie'
	config = {
	'user_file_1':path + 'user_ids_' + args.source + '.csv', #
	'item_file_1':path + 'item_ids_' + args.source + '.csv',
	'user_file_2':path + 'user_ids_' + args.target + '.csv',
	'item_file_2':path + 'item_ids_' + args.target + '.csv',
	'user_item_npy_1':path + 'user_item_dict_' + args.source + '.npy', # dict, key是user id, value 是该user 连接的 positive items (CD domain, source domain)
	'user_item_npy_2':path + 'user_item_dict_' + args.target + '.npy', # dict , key是user id, value 是该user 连接的 positive items (Movie domain, target domain)
	'user_item_pair_npy_1':path + 'user_item_pair_dict_' + args.source + '.npy', # dict, key 是user id，value是该user和其所有 positive items组成的pair，例如： user1:[[user1,pos_item1],[user1,pos_item2]]
	'user_item_pair_npy_2':path + 'user_item_pair_dict_' + args.target + '.npy',
	'cold_user_ids_file':path + 'cold_user_ids_' + args.source + '_and_' + args.target + '_cold' + str(args.cold_ratio) +'.csv', # 从overlapped user中选出部分用于测试集中的冷启动用户
	'warm_user_ids_file':path + 'warm_user_ids_' + args.source + '_and_' + args.target + '_cold' + str(args.cold_ratio) +'.csv', # overlapped user 中去除cold-start user之后的user
	'source_name':args.source,
	'target_name':args.target
	}
	# 准备数据
	# 数据编号顺序： shared warm users -> shared cold start users -> unshared users -> items
	start = datetime.datetime.now()
	# load source domain data, 这里的trainX 和 valX用于单个domain的prediction,trainX[:,0]=userIdx, trainX[:,1]=posIdx, trainX[:,2]=negIdx
	num_userX, num_itemX, GX, warm_user_ids, cold_user_ids, unshared_uidsX, train_user_item_dictX, trainX = load(config['user_file_1'],config['item_file_1'],config['warm_user_ids_file'],config['cold_user_ids_file'],config['user_item_npy_1'],config['user_item_pair_npy_1'],domain='source')
	# load target domain data
	num_userY, num_itemY, GY, warm_user_ids, cold_user_ids, unshared_uidsY, train_user_item_dictY, test_user_item_dict,trainY, train_cross_dict, val_cross_dict = load(config['user_file_2'],config['item_file_2'],config['warm_user_ids_file'],config['cold_user_ids_file'],config['user_item_npy_2'],config['user_item_pair_npy_2'],domain='target')

	end = datetime.datetime.now()
	cprint('Prepare data cost: '+ str(end-start), 'yellow')
	n_warm = len(warm_user_ids)
	n_cold = len(cold_user_ids)
	path_emb = 'path to input embed'
	if args.user_emb_mode == 'mean':
		user_emb_X = np.load(path_emb + 'CDs_and_Vinyl_mean_user_emb.npy')
		item_emb_X = np.load(path_emb + 'CDs_and_Vinyl_description_emb.npy')
		user_emb_Y = np.load(path_emb + 'Movies_and_TV_mean_user_emb.npy')
		item_emb_Y = np.load(path_emb + 'Movies_and_TV_description_emb.npy')
	elif args.user_emb_mode == 'sum':
		user_emb_X = np.load(path_emb + 'CDs_and_Vinyl_sum_user_emb.npy')
		item_emb_X = np.load(path_emb + 'CDs_and_Vinyl_description_emb.npy')
		user_emb_Y = np.load(path_emb + 'Movies_and_TV_sum_user_emb.npy')
		item_emb_Y = np.load(path_emb + 'Movies_and_TV_description_emb.npy')
	elif args.user_emb_mode == 'random':
		user_emb_X = np.random.randn(num_userX, 64)
		item_emb_X = np.random.randn(num_itemX, 64)
		user_emb_Y = np.random.randn(num_userY, 64)
		item_emb_Y = np.random.randn(num_itemY, 64)	
	else:
		cprint('Loading pretrained embedding...')
		if args.source == 'CD':
			path_emb = 'path to pretrained cd2movie emb'
			user_emb_X = np.load(path_emb + 'user_CDs_and_Vinyl.npy')
			item_emb_X = np.load(path_emb + 'item_CDs_and_Vinyl.npy')
			user_emb_Y = np.load(path_emb + 'user_Movie_and_TV.npy')
			item_emb_Y = np.load(path_emb + 'item_Movie_and_TV.npy')
		elif args.source=='Book':

			path_emb = 'path to pretrained book2cd emb'
			user_emb_X = np.load(path_emb + 'user_book.npy')
			item_emb_X = np.load(path_emb + 'item_book.npy')
			user_emb_Y = np.load(path_emb + 'user_cd.npy')
			item_emb_Y = np.load(path_emb + 'item_cd.npy')		

	featureX = np.concatenate((user_emb_X,item_emb_X),0)
	# target domain 中把cold-start user拿掉
	featureY = np.concatenate((user_emb_Y[:n_warm,:], user_emb_Y[(n_warm+n_cold):,:], item_emb_Y),0)
	featureX_cuda = torch.FloatTensor(featureX).cuda()
	featureY_cuda = torch.FloatTensor(featureY).cuda()
	if args.normalize:
		featureX_cuda = F.normalize(featureX_cuda, dim=1)
		featureY_cuda = F.normalize(featureY_cuda, dim=1)

	n_user_uniqueX = len(unshared_uidsX) + len(cold_user_ids) #
	n_user_uniqueY = len(unshared_uidsY)
	n_shared = n_shared = len(warm_user_ids)
	in_dimX = user_emb_X.shape[1]
	in_dimY = user_emb_Y.shape[1]

	GX_dgl = dgl.DGLGraph()
	GX_dgl.from_networkx(GX)
	GY_dgl = dgl.DGLGraph()
	GY_dgl.from_networkx(GY)
	
	encoderX2Y = Encoder(GX_dgl, featureX_cuda, in_dimX, args.hidden, args.out_dim_encode, args.n_layers, F.leaky_relu_, 0.0).cuda()
	encoderY2X = Encoder(GY_dgl, featureY_cuda, in_dimY, args.hidden, args.out_dim_encode, args.n_layers, F.leaky_relu_, 0.0).cuda()
	in_dim = args.out_dim_encode

	out_dimX_1 = args.hidden[-1] 
	out_dimX_2 = in_dimY 
	out_dimY_1 = args.hidden[-1]
	out_dimY_2 = in_dimX
	decoderX2Y = Decoder(in_dim, out_dimX_1, out_dimX_2).cuda()
	decoderY2X = Decoder(in_dim, out_dimY_1, out_dimY_2).cuda()
		
	in_dim_disX = in_dimX + out_dimX_2
	in_dim_disY = in_dimY + out_dimY_2

	optim_pre_X = torch.optim.Adam(encoderX2Y.parameters(), lr=args.lr, betas=(args.beta, 0.999))
	optim_pre_Y = torch.optim.Adam(encoderY2X.parameters(), lr=args.lr, betas=(args.beta, 0.999))

	optim_pre_X2Y = torch.optim.Adam(itertools.chain(encoderX2Y.parameters(), decoderX2Y.layers.parameters()), lr=args.lr,betas=(args.beta,0.999),weight_decay=args.decay)
	optim_pre_Y2X = torch.optim.Adam(itertools.chain(encoderY2X.parameters(), decoderY2X.layers.parameters()), lr=args.lr,betas=(args.beta,0.999),weight_decay=args.decay)
	optim_auto_X = torch.optim.Adam(itertools.chain(encoderX2Y.parameters(),decoderY2X.parameters()), lr=args.lr, betas=(args.beta,0.999),weight_decay=args.decay)
	optim_auto_Y = torch.optim.Adam(itertools.chain(encoderY2X.parameters(),decoderX2Y.parameters()), lr=args.lr, betas=(args.beta,0.999),weight_decay=args.decay)
	optim_superX2Y = torch.optim.Adam(itertools.chain(encoderX2Y.parameters(),decoderX2Y.parameters()), lr=args.lr, betas=(args.beta,0.999),weight_decay=args.decay)
	optim_superY2X = torch.optim.Adam(itertools.chain(encoderY2X.parameters(),decoderY2X.parameters()), lr=args.lr, betas=(args.beta,0.999),weight_decay=args.decay)
	optims = [optim_pre_X, optim_pre_Y, optim_pre_X2Y, optim_pre_Y2X, optim_auto_X, optim_auto_Y, optim_superX2Y,optim_superY2X]
	if args.load_model:
		encoderX2Y.load_state_dict(torch.load('./models/encoderX2Y.pkl')) 
		encoderY2X.load_state_dict(torch.load('./models/encoderY2X.pkl')) 
		decoderX2Y.load_state_dict(torch.load('./models/decoderX2Y.pkl'))
		decoderY2X.load_state_dict(torch.load('./models/decoderY2X.pkl'))
	save_encoderX2Y_name = './models/encoderX2Y.pkl'
	save_encoderY2X_name = './models/encoderY2X.pkl'
	save_decoderX2Y_name = './models/decoderX2Y.pkl'
	save_decoderY2X_name = './models/decoderY2X.pkl'
	result_path = './result/' + config['source_name'] + '2'+ config['target_name'] + '/'
	result_file = result_path + 'DAN_' + datetime.datetime.now().strftime('%Y-%m-%d')+'.txt'
	if not os.path.exists(result_path):
		os.makedirs(result_path)
	ofile = open(result_file,'a')
	ofile.write('Time: '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'\n')
	ofile.write('Config: \n')
	for key, value in vars(args).items():
		ofile.write('%s:%s\n'%(key, value))
		print(key, value)

	##### Training #####
	print('Single-domain X prediction')
	batchSize = args.batchSize
	batch_links = args.batch_links
	stopping_step = 0
	flag_step = 20
	cur_best_ndcg = 0.0

	for epoch in range(args.n_epochs):
		random.shuffle(warm_user_ids)
		print('Epoch: ',epoch)
		start = datetime.datetime.now()		
		
		print('Single-domain X prediction')
		n_batchX = len(trainX) // batch_links + 1
		print('{:d} batchs in domain X'.format(n_batchX))

		for batch in range(n_batchX):
			batchX = trainX[batch*batch_links:(batch+1)*batch_links,:]
			userIdx_X = torch.LongTensor(batchX[:,0]).cuda()
			pos_itemIdx_X = torch.LongTensor(batchX[:,1]).cuda()
			neg_itemIdx_X = torch.LongTensor(batchX[:,2]).cuda()
			loss_X,loss_reg_X, loss_pre_X = train_unshared(args, encoder=encoderX2Y,optim_pre=optim_pre_X ,userIdx=userIdx_X, posIdx=pos_itemIdx_X, negIdx=neg_itemIdx_X, reg=args.bpr_reg)
			writer.add_scalars('Train Loss',{'loss_X':loss_X}, epoch*n_batchX + batch)
			if batch % 400 == 0:
				print('batch {:d},loss_X: {:.4f} | loss_reg_X: {:.4f} | Loss_pre_X : {:.4f}'.format(batch,loss_X,loss_reg_X,loss_pre_X))
			gc.collect()
		
		print('Single-domain Y prediction')
		n_batchY = len(trainY) // batch_links + 1
		print('{:d} batchs in domain Y'.format(n_batchY))
		for batch in range(n_batchY):
			batchY = trainY[batch*batch_links:(batch+1)*batch_links,:]
			userIdx_Y = torch.LongTensor(batchY[:,0]).cuda()
			pos_itemIdx_Y = torch.LongTensor(batchY[:,1]).cuda()
			neg_itemIdx_Y = torch.LongTensor(batchY[:,2]).cuda()
			loss_Y,loss_reg_Y, loss_pre_Y = train_unshared(args, encoder=encoderY2X,optim_pre=optim_pre_Y, userIdx=userIdx_Y, posIdx=pos_itemIdx_Y, negIdx=neg_itemIdx_Y,reg=args.bpr_reg)
			writer.add_scalars('Train Loss',{'loss_Y':loss_Y},epoch*n_batchY + batch)
			if batch % 200 == 0:
				print('batch: {:d}, loss_Y: {:.4f}| loss_reg_Y: {:.4f} | loss_pre_Y : {:.4f}'.format(batch,loss_Y,loss_reg_Y,loss_pre_Y))
			gc.collect()
		
		# shared user
		n_batch = n_warm // batchSize + 1
		print('{:d} batchs in cross domain'.format(n_batch))
		for batch in range(n_batch):
			batch_users = warm_user_ids[batch* batchSize : (batch+1) * batchSize]
			batch_users = np.array(batch_users)
			inputsX = featureX[batch_users,:]
			inputsY = featureY[batch_users,:]
			batch_users = torch.LongTensor(batch_users).cuda()
			inputsX_cuda = torch.FloatTensor(inputsX).cuda()
			inputsY_cuda = torch.FloatTensor(inputsY).cuda()

			loss_all, loss_preX, loss_regX,loss_preY,loss_regY,loss_preX2Y, loss_preY2X,loss_regX2Y,loss_regY2X,\
			loss_autoX, loss_autoY,loss_superX2Y,loss_superY2X = train_shared(args, optims,inputsX_cuda, inputsY_cuda,\
														 	encoderX2Y, encoderY2X, decoderX2Y, decoderY2X,\
															batch_users, train_user_item_dictX,\
															train_user_item_dictY, train_cross_dict,reg=args.bpr_reg)
			writer.add_scalars('Train Loss',{'loss_preX': loss_preX, 'loss_preY': loss_preY,'loss_preX2Y':loss_preX2Y,'loss_preY2X':loss_preY2X,'loss_regX':loss_regX,'loss_regY':loss_regY,'loss_regX2Y':loss_regX2Y,'loss_regY2X':loss_regY2X,'loss_all':loss_all,'loss_autoX':loss_autoX,'loss_autoY':loss_autoY,'loss_superX2Y':loss_superX2Y,'loss_superY2X':loss_superY2X},epoch*n_batch+batch)
			if batch % 80 == 0:
				print('batch {:d} \n loss_all: {:.4f} | loss_preX2Y: {:.4f} | loss_preY2X: {:.4f}\n loss_regX2Y:{:.4f} | loss_regY2X: {:.4f} \n loss_autoX: {:.4f} | loss_autoY: {:.4f} \n loss_superX2Y:{:.4f} | loss_superY2X: {:.4f}'.format(batch, loss_all, loss_preX2Y, loss_preY2X,loss_regX2Y,loss_regY2X, loss_autoX, loss_autoY, loss_superX2Y,loss_superY2X))

			gc.collect()
		if epoch==0:
			end = datetime.datetime.now()
			cprint('Training one epoch cost: ' + str(end - start), 'yellow', attrs=['bold'])
		# evaluation
		start = datetime.datetime.now()
		warm_users_cuda = torch.LongTensor(warm_user_ids).cuda()
		hit_rate, ndcg = evaluate(args, encoderX2Y, encoderY2X, decoderX2Y, warm_users_cuda, val_cross_dict)
		if epoch == 0:
			end = datetime.datetime.now()
			cprint('Evaluating one epoch cost: '+ str(end - start), 'yellow', attrs=['bold'])
		for i,k in enumerate(args.topk):
			print('Hit@{:d}: {:.4f} | NDCG@{:d}: {:.4f}'.format(k, hit_rate[i], k, ndcg[i]))
		# early stopping
		cur_best_ndcg, stopping_step, should_stop = early_stopping(ndcg[2], cur_best_ndcg, stopping_step, flag_step)
		# 保存取得最好性能时候的模型
		if ndcg[2] == cur_best_ndcg:
			torch.save(encoderX2Y.state_dict(), save_encoderX2Y_name)
			torch.save(encoderY2X.state_dict(), save_encoderY2X_name)
			torch.save(decoderX2Y.state_dict(), save_decoderX2Y_name)
			torch.save(decoderY2X.state_dict(), save_decoderY2X_name)

		if should_stop:
			cprint('Early stopping at epoch '+ str(epoch), 'magenta')
			print('Early stopping at epoch ', epoch, file=ofile)
			cprint('Best ndcg@20: ' + str(cur_best_ndcg), 'magenta')
			print('Best ndcg@20: ',cur_best_ndcg, file=ofile)
			break
	# test
	if args.test:
		start = datetime.datetime.now()
		cprint('Testing...','yellow')
		print('Testing...', file=ofile)
		encoderX2Y.load_state_dict(torch.load(save_encoderX2Y_name)) 
		encoderY2X.load_state_dict(torch.load(save_encoderY2X_name)) 
		decoderX2Y.load_state_dict(torch.load(save_decoderX2Y_name))

		cold_users_cuda = torch.LongTensor(cold_user_ids).cuda()
		hit_rate, ndcg = evaluate(args, encoderX2Y, encoderY2X, decoderX2Y, cold_users_cuda, test_user_item_dict,test=True)
		end = datetime.datetime.now()
		print('Testing cost: ',end - start)
		for i,k in enumerate(args.topk):
			print('Hit@{:d}: {:.4f} | NDCG@{:d}: {:.4f}'.format(k, hit_rate[i], k, ndcg[i]))
			print('Hit@{:d}: {:.4f} | NDCG@{:d}: {:.4f}'.format(k, hit_rate[i], k, ndcg[i]), file=ofile)
	ofile.write('==================================\n\n')
	ofile.close()
