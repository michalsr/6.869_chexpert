import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from tqdm  import tqdm
import time
from data.dataset import ChestDataSet
from models.densenet import densenet121
from config import opt
from data import preprocess
from sklearn.metrics import roc_auc_score
def train(**kwargs):
	model = generate_model()
	train_data = ChestDataSet(opt.data_root, opt.train_data_list, mode='train')
	train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)
	val_data = ChestDataSet(opt.data_root, opt.valid_data_list,
			mode = 'train')
	val_dataloader = DataLoader(val_data, opt.batch_size,
			shuffle = False)
	criterion = torch.nn.BCELoss(reduction='mean')
	optimizer = torch.optim.Adam(model.parameters(), betas=opt.betas,
			lr=opt.lr, eps=opt.eps)
	loss_mean_min = 1e100
	print('\n-------------------')
	print('-Start training....')
	print('-----------------\n')
	for epoch in range(opt.max_epoch):
		print('- Epoch', epoch+1)
		model.train()
		total_batch = int(len(train_data)/opt.batch_size)
		bar = tqdm(enumerate(train_dataloader),total=total_batch)
		for i, (data,label) in bar:
			torch.set_grad_enabled(True)
			inp = data.clone().detach().requires_grad_(True)
			target = label.clone().detach()
			inp = inp.cuda()
			target = target.cuda()
			optimizer.zero_grad()
			output = model(inp)
			loss = criterion(output,target)
			loss.backward()
			optimizer.step()
			bar.set_postfix_str('loss: %.5s' % loss.item())
		loss_mean = val(model,val_dataloader, criterion,total_batch)
		time_end=time.strftime('%m%d_%H%M%S')
		if loss_mean_min > loss_mean:
			loss_mean_min = loss_mean
			torch.save({'epoch':epoch+1,
				    'state_dict':model.state_dict(),
				    'optimizer': optimizer.state_dict()},
				    '/checkpoints/m_' + '.pth.tar')
			print('Epoch [' + str(epoch+1) + '] [save] [m_' + 
				time_end + '] loss = ' +str(loss_mean))
		else:
			print('Epoch [' + str(epoch+1) + '] [-----] [m_' +
				time_end + '] loss =' + str(loss_mean))
def val(model,dataloader, criterion, total_batch):
	model.eval()
	counter = 0
	loss_sum = 0
	with torch.no_grad():
		bar = tqdm(enumerate(datalaoder),total=total_batch)
		for i , (data,label) in bar:
			inp=data.clone().detach()
			target=label.clone().detach()
			inp = inp.cuda()
			target =target.cuda()
			output = model(inp)
			loss = criterion(output,target)
			loss_sum += loss.item()
			counter +=1
			bar.set_postfix_str('loss: %.5s' % loss.item())
	loss_mean = loss_sum/counter
	return loss_mean
	

def generate_model():
	model = densenet121(len(opt.classes))
	model.cuda()
	if opt.load_model_path:
		load_model_path = os.path.join('./checkpoints',	
			opt.load_model_path)
		print('Loading checkpoint ....')
		checkpoint = torch.load(load_model_path)
		model.load_state_dict(checkpoint['state_dict'])
		print('Done')
	return model


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--run_type',type=str,required=True,
				help='train or test')
	args=parser.parse_args()
	if args.run_type == 'train':
		preprocess.preprocess()
		train()
