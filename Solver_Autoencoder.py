from numpy import isnan, inf
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
	def __init__(self, args, params, CViter):
		torch.cuda.empty_cache() 
		self.args = args
		self.params = params
		self.CViter = CViter
		self.dataloaders = fetch_dataloader(['train', 'val'], params, CViter) 
		self.model = model_loader.loadModel(params, netname = args.network, dropout_rate = params.dropout_rate).cuda()
		self.optimizer = torch.optim.Adam(	self.model.parameters(), 
							params.learning_rate, 
							betas=(0.9, 0.999), 
							eps=1e-08, 
							weight_decay=params.weight_decay, 
							amsgrad=False)
		self.loss_fn = L2_loss
		
		
	def __step__(self):
		torch.cuda.empty_cache() 
		logging.info("Training")
		losses = AverageMeter()
		# switch to train mode
		self.model.train()
		loss = []
		print('**********TRAINING**********')
		with tqdm(total=len(self.dataloaders['train'])) as t:
			for i, (datas, label) in enumerate(self.dataloaders['train']):
				logging.info("        Loading Varable")
				# compute output
				logging.info("        Compute output")
				output = self.model(torch.autograd.Variable(datas.cuda())).double()

				# measure record cost
				cost = self.loss_fn(output, torch.autograd.Variable(datas.cuda()))
				assert not isnan(cost.cpu().data.numpy().any()),  "Gradient exploding, Loss = {}".format(cost.cpu().data.numpy())
				losses.update(cost.cpu().data.numpy(), len(datas))
				
				del output

				# compute gradient and do SGD step
				logging.info("        Compute gradient and do SGD step")
				self.optimizer.zero_grad()
				cost.backward()
				self.optimizer.step()
			
				gc.collect()
				t.set_postfix(loss='{:05.3f}'.format(losses()))
				t.update()

		return loss
	
	
	def validate(self):
		torch.cuda.empty_cache() 
		logging.info("Validating")
		losses = AverageMeter()
		# switch to evaluate mode
		self.model.eval()
		print('----------VALIDATING----------')
		with tqdm(total=len(self.dataloaders['val'])) as t:
			for i, (datas, label) in (enumerate(self.dataloaders['val'])):
				logging.info("        Compute output")
				output = self.model(torch.autograd.Variable(datas.cuda())).double()
				
				logging.info("        Computing loss")
				loss = self.loss_fn(output, torch.autograd.Variable(datas.cuda()))
				assert not isnan(loss.cpu().data.numpy()),  "Overshot loss, Loss = {}".format(loss.cpu().data.numpy())
				
				# measure record cost
				losses.update(loss.cpu().data.numpy(), len(datas))
				
				del output
				
				gc.collect()
				t.update()

		return losses()


	def train(self):
		start_epoch = 0
		best_val_loss = inf	

		if self.args.resume:
			logging.warning('Resuming Checkpoint')
			start_epoch, best_val_loss = self.__resume_checkpoint__('')
			if not start_epoch < self.params.epochs:
				logging.warning('Skipping training for finished model\n')
				return []			
			
		logging.warning('    Starting With Best loss = {loss:.4f}'.format(loss = best_val_loss))
		logging.warning('Initialize training from {} to {} epochs'.format(start_epoch, self.params.epochs))

		for epoch in range(start_epoch, self.params.epochs):
			logging.warning('CV [{}], Training Epoch: [{}/{}]'.format('_'.join(tuple(map(str, self.CViter))), epoch+1, self.params.epochs))

			self.__step__()
			gc.collect()

			# evaluate on validation set
			val_loss= self.validate()
			gc.collect()

			# remember best loss and save checkpoint
			logging.warning('    Loss {loss:.4f};\n'.format(loss=val_loss))		
			if val_loss < best_val_loss:
				self.__save_checkpoint__({
					'epoch': epoch + 1,
					'state_dict': self.model.state_dict(),
					'loss': val_loss,
					'optimizer' : self.optimizer.state_dict(),
					}, 'best')
				best_val_loss = val_loss
				logging.warning('    Saved Best AUC model with loss \n{} \n'.format(val_loss))

			self.__save_checkpoint__({
					'epoch': epoch + 1,
					'state_dict': self.model.state_dict(),
					'loss': best_val_loss,
					'optimizer' : self.optimizer.state_dict(),
					}, '')
			if epoch % 7 == 6 and epoch > 0:
				print('*****************decay**************')
				self.__learning_rate_decay__(self.optimizer, self.args.lrDecay)

		gc.collect()
		logging.warning('Training finalized\n')
		return
		
	def __save_checkpoint__(self, state, checkpoint_type):
		checkpointpath, checkpointfile = get_checkpointname(	self.args, 
									self.params.num_LatentFeatures, 
									checkpoint_type, 
									self.CViter)
		if not os.path.isdir(checkpointpath):
			os.mkdir(checkpointpath)
			
		torch.save(state, checkpointfile)


	def __resume_checkpoint__(self, checkpoint_type):
		_, checkpointfile = get_checkpointname(self.args, self.params.num_LatentFeatures, checkpoint_type, self.CViter)
		
		if not os.path.isfile(checkpointfile):
			return 0, inf
		else:
			logging.info("Loading checkpoint {}".format(checkpointfile))
			checkpoint = torch.load(checkpointfile)
			start_epoch = checkpoint['epoch']
			loss = checkpoint['loss']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			
			if loss == 0:
				loss = inf
				
			return start_epoch, loss

	def __learning_rate_decay__(self, optimizer, decay_rate):
		if decay_rate < 1:
			for param_group in optimizer.param_groups:
				param_group['lr'] = param_group['lr'] * decay_rate
