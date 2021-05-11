from numpy import isnan
import torch
import logging
import gc

import torch.backends.cudnn as cudnn
from Evaluation_Matix import *
from utils import *
import model_loader
from data_loader import fetch_dataloader
from tqdm import tqdm
from datetime import datetime

class Solver:
	def __init__(self, args, params, CViter, test = False):
		self.args = args
		self.params = params
		self.CViter = CViter
		if test:
			self.dataloaders = fetch_dataloader(['test'], params, CViter) 
		else:
			self.dataloaders = fetch_dataloader(['train', 'val'], params, CViter) 

		self.loss_fn = L2_loss
		
		
	def __step__(self):
		#self.model(data, latent = True) to retrieve latent features

		pass
	
	
	def validate(self):

		pass


	def train(self):
		start_epoch = 0
		best_val_loss = 0	

		logging.warning('Resuming Checkpoint')
		_, best_val_loss = self.__resume_checkpoint__('best')
		

		for epoch in range(1, self.params.epochs):

			self.__step__()
			gc.collect()

			# evaluate on validation set
			val_loss = self.validate()
			gc.collect()


			self.__save_checkpoint__({
					'epoch': epoch + 1,
					'state_dict': self.model.state_dict(),
					'loss': best_val_loss,
					'optimizer' : self.optimizer.state_dict(),
					}, 'Clustered')


		gc.collect()
		logging.warning('Training finalized\n')
		return
		
	def __save_checkpoint__(self, state, itemindex):
		checkpointfile = os.path.join(self.args.model_dir, self.args.network)
		checkpointfile = os.path.join(checkpointfile, 'Checkpoints' + str(itemindex))
		if not os.path.isdir(checkpointfile):
			os.mkdir(checkpointfile)
		checkpointfile = os.path.join(checkpointfile, 
					'{network}_{LrD}_{cv_iter}.pth.tar'.format(network = self.args.network, 
										     LrD = self.args.lrDecay,
										     cv_iter = '_'.join(tuple(map(str, self.CViter)))))
		torch.save(state, checkpointfile)


	def __resume_checkpoint__(self, itemindex):
		checkpointfile = os.path.join(self.args.model_dir, self.args.network)
		checkpointfile = os.path.join(checkpointfile, 'Checkpoints' + str(itemindex))
		checkpointfile = os.path.join(checkpointfile, 
					'{network}_{LrD}_{cv_iter}.pth.tar'.format(network = self.args.network, 
										     LrD = self.args.lrDecay,
										     cv_iter = '_'.join(tuple(map(str, self.CViter)))))
		if not os.path.isfile(checkpointfile):
			return 0, 0
		else:
			logging.info("Loading checkpoint {}".format(checkpointfile))
			checkpoint = torch.load(checkpointfile)
			start_epoch = checkpoint['epoch']
			loss = checkpoint['loss']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			
		return start_epoch, loss

	def __learning_rate_decay__(self, optimizer, decay_rate):
		if decay_rate < 1:
			for param_group in optimizer.param_groups:
				param_group['lr'] = param_group['lr'] * decay_rate
