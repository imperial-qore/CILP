import pickle
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from .models import *

def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	data = torch.tensor(data)
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w)
	return torch.stack(windows)

def normalize(data, minv, maxv):
	return (data - minv) / (maxv - minv)

def denormalize(data, minv, maxv):
	return (data * (maxv - minv)) + minv

def load_dataset(dataset):
	fname = base_url + f'datasets/{dataset}'
	dset = np.abs(np.genfromtxt(fname, delimiter=','))
	minv, maxv = np.min(dset), np.max(dset)
	dset = normalize(dset, minv, maxv)
	split = int(0.9 * dset.shape[0])
	train, test = dset[:split], dset[split:]
	return train, test, minv, maxv

def save_model(model, optimizer, scheduler, epoch, loss_list):
	folder = base_url + f'checkpoints/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/{model.name}.ckpt'
	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'scheduler_state_dict': scheduler.state_dict(),
		'loss_list': loss_list}, file_path)

def load_model(modelname, dims):
	import provisioner.src.models
	model_class = getattr(provisioner.src.models, modelname)
	model = model_class(dims).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	folder = base_url + f'checkpoints/'
	fname = f'{folder}/{model.name}.ckpt'
	if os.path.exists(fname):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		loss_list = checkpoint['loss_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; loss_list = []
	return model, optimizer, scheduler, epoch, loss_list

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
	feats = dataO.shape[1]; ls = []
	l = nn.MSELoss(reduction = 'mean')
	for i, d in enumerate(data):
		gold = data[i+1][-1] if i+1 < data.shape[0] else data[i][-1]
		if 'LSTM' in model.name:
			pred = model(d)
		elif 'Attention' in model.name:
			pred = model(d)
		loss = l(pred, gold)
		ls.append(torch.mean(loss).item())
		if training:
			optimizer.zero_grad(); loss.backward(); optimizer.step()
	if training: scheduler.step()
	return np.mean(ls), optimizer.param_groups[0]['lr']

def plot_accuracies(loss_list, folder, model):
	os.makedirs(f'{folder}/plots/', exist_ok=True)
	trainAcc = [i[0] for i in loss_list]
	testAcc = [i[1] for i in loss_list]
	lrs = [i[1] for i in loss_list]
	plt.xlabel('Epochs')
	plt.ylabel('Average Loss')
	plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', color='k', linewidth=1, linestyle='solid', marker='.')
	plt.plot(range(len(testAcc)), testAcc, label='Average Testing Loss', color='b', linewidth=1, linestyle='dashed', marker='.')
	plt.twinx()
	plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='dotted', marker='.')
	plt.savefig(f'{folder}/plots/{model.name}.pdf')
	plt.clf()

def freeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = False

def unfreeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = True

class color:
	HEADER = '\033[95m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	RED = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'