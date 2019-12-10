import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from tqdm  import tqdm
import time
from data.dataset import ChestDataSet
from models.densenet import densenet121
from models.aecnn0 import AECNN0
from config import opt
from data import preprocess
from sklearn.metrics import roc_auc_score
def test(**kwargs):
	model =generate_model()
	test_data = ChestDataSet(opt.data_root, opt.test_data_list,mode='train')
	test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
	total_batch = int(len(test_data)/opt.batch_size)
	gt = torch.FloatTensor()
	pred = torch.FloatTensor()
	gt = gt.cuda()
	pred = pred.cuda()
	print('\n--------------------')
	print('Start Testing')
	model.eval()
	with torch.no_grad():
		bar=tqdm(enumerate(test_dataloader),total=total_batch)
		for i, (data,label) in bar:
			bt, c, w, h = data.shape
			data = data.mean(1).reshape(bt, 1, w, h)
			inp = data.clone().detach()
			target = label.clone().detach()
			inp = inp.cuda()
			target = target.cuda()
			output = model(inp)
			gt = torch.cat((gt,target),0)
			#print(gt)
			pred = torch.cat((pred, output[1].data),0)
	AUROCs= compute_AUCs(gt,pred)
	AUROC_avg = np.array(AUROCs).mean()
	print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
	for i in range(len(opt.classes)):
		print('The AUROC of {} is {}'.format(opt.classes[i],AUROCs[i]))
	write_csv(AUROCs,opt.result_file)
def train(**kwargs):
	model = generate_model()
	train_data = ChestDataSet(opt.data_root, opt.train_data_list, mode='train')
	train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)
	val_data = ChestDataSet(opt.data_root, opt.valid_data_list,
			mode = 'train')
	val_dataloader = DataLoader(val_data, opt.batch_size,
			shuffle = False)
	#criterion = torch.nn.BCELoss(reduction='mean')
	loss1 = torch.nn.MSELoss(size_average = True)
	loss2 = torch.nn.BCELoss(size_average = True)

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
			bt, c, w, h = data.shape
			data = data.mean(1).reshape(bt, 1, w, h)
			
			varInput = data.clone().detach().requires_grad_(True).cuda()
			varTarget1 = data.cuda()
			varTarget2 = label.clone().detach().cuda()

			varOutput1, varOutput2 = model(varInput)
			lossvalue1 = loss1(varOutput1, varTarget1)
			lossvalue2 = loss2(varOutput2, varTarget2)
			loss = 0.1*lossvalue1 + 0.9*lossvalue2

			# inp = data.clone().detach().requires_grad_(True)
			# target = label.clone().detach()
			
			# inp = inp.cuda()
			# target = target.cuda()

			optimizer.zero_grad()
			# output = model(inp)

			# loss = criterion(output[1], target)

			loss.backward()
			optimizer.step()
			bar.set_postfix_str('loss: %.5s' % loss.item())
		loss_mean = val(model,val_dataloader,total_batch)
		time_end=time.strftime('%m%d_%H%M%S')
		if loss_mean_min > loss_mean:
			loss_mean_min = loss_mean
			save_dir = 'aecnn0_bt_64'
			if not os.path.exists(save_dir):
				os.mkdir(save_dir)
			PATH=save_dir+"/weights"+str(epoch)
			torch.save(model.state_dict(), PATH)
			print('Epoch [' + str(epoch+1) + '] [save] [m_' + 
				time_end + '] loss = ' +str(loss_mean))
		else:
			print('Epoch [' + str(epoch+1) + '] [-----] [m_' +
				time_end + '] loss =' + str(loss_mean))
def val(model,dataloader, total_batch):
	model.eval()
	counter = 0
	loss_sum = 0

	loss1 = torch.nn.MSELoss(size_average = True)
	loss2 = torch.nn.BCELoss(size_average = True)

	with torch.no_grad():
		bar = tqdm(enumerate(dataloader),total=total_batch)
		for i , (data,label) in bar:
			# inp=data.clone().detach()
			# target=label.clone().detach()
			# inp = inp.cuda()
			# target =target.cuda()
			# output = model(inp)
			# loss = criterion(output,target)

			bt, c, w, h = data.shape
			data = data.mean(1).reshape(bt, 1, w, h)

			varInput = data.clone().detach().requires_grad_(True).cuda()
			varTarget1 = data.cuda()
			varTarget2 = label.clone().detach().cuda()

			# varInput dim: bsx1x896x896
			varOutput1, varOutput2 = model(varInput)
			lossvalue1 = loss1(varOutput1, varTarget1)
			lossvalue2 = loss2(varOutput2, varTarget2)
			loss = 0.1*lossvalue1 + 0.9*lossvalue2

			loss_sum += loss.item()
			counter +=1
			bar.set_postfix_str('loss: %.5s' % loss.item())
	loss_mean = loss_sum/counter
	return loss_mean
def compute_AUCs(gt,pred):
	''' Computes Area Under the Curve (AUC) from prediction scores.
	    Args:
		gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
		true binary labels
		pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
			can be either be probability estimates of the
			 positive class, confidence values or binary
			decisions.
	   Returns:
		List of AUROCs of all classes
	'''
	AUROCs = []
	gt_np = gt.cpu().numpy()
	
	pred_np = pred.cpu().numpy()
	#print(pred_np)
	#print(roc_auc_score(gt_np,pred_np))
	for i in range(len(opt.classes)):
		
		#print(gt_np[:,i])
		AUROCs.append(roc_auc_score(gt_np[:,i],pred_np[:,i]))
	return AUROCs
def write_csv(results,file_name):
	df = pd.DataFrame(results)
	df.to_csv(file_name, sep=' ')


def generate_model():
	model = AECNN0(len(opt.classes))
	model.cuda()
	if opt.load_model_path:

		print('Loading checkpoint ....')
		model.load_state_dict(torch.load(opt.load_model_path))
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
	if args.run_type == 'test':
		#preprocess.preprocess()
		test()	
