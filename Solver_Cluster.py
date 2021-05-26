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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score
import numpy as np

class Solver:
	def __init__(self, args, params, CViter, test = False):
		self.args = args
		self.params = params
		self.CViter = CViter
		self.model = model_loader.loadModel(params, netname = args.network).cuda()
		if test:
			self.__resume_checkpoint__('Clustered') 
			self.dataloaders = fetch_dataloader(['test'], params, CViter) 
		else:
			self.__resume_checkpoint__('best')  # load the best performing autoencoder
			self.dataloaders = fetch_dataloader(['train', 'val'], params, CViter) 

		self.loss_fn = L2_loss
		self.kmean = MiniBatchKMeans(n_clusters = 2, max_iter=1000, batch_size=1000, reassignment_ratio = 1, n_init = 50)
		
	def __infer_cluster_labels__(self, actual_labels):
		'''
		A very rough algorithm to infer which cluster is which label
		(warning: not generalizable for other dataset than breast pathology)

		args:
		kmeans: kmeans object (contain result)
		actual_labels: y
		return:
		dictionary of {label: cluster}
		'''
		labels = {}
		counts = {}

		for i in range(self.kmean.n_clusters):

			# find index of points in cluster (index for cluster 0 or cluster 1)

			#                  this label is clustering label
			index = np.where(self.kmean.labels_ == i)

			# append actual labels for each point in cluster
			# extract all true labels on cluster i
			labels[i] =actual_labels[index]
			counts[i] = np.bincount(labels[i])

		inferred_labels = {}

		inferred_labels[0] = []
		inferred_labels[1] = []

		if counts[0][0]/sum(counts[0]) >= counts[1][0]/sum(counts[1]):
			inferred_labels[0].append(0)
		else:
			inferred_labels[0].append(1)

		if counts[0][1]/sum(counts[0]) >= counts[1][1]/sum(counts[1]):
			inferred_labels[1].append(0)
		else:
			inferred_labels[1].append(1)

		return inferred_labels
		
	def __infer_data_labels__(self, X_labels):
		'''
		infer data's label by cluster's label

		args: 
		X_labels:  list of predicted clusters of X
		cluster_labels: dictionary of {label: cluster}

		return:
		list of predicted labels, same length of X
		'''
		# empty array of len(X)
		predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

		for i, cluster in enumerate(X_labels):
			for key, value in self.cluster_labels.items():
				if cluster in value:
					predicted_labels[i] = key

		return np.array(predicted_labels)
		
		
	def __step__(self):
		x = []
		y = []
		print('training')
		with tqdm(total=len(self.dataloaders['train'])) as t:
			for i, (data, label) in enumerate(self.dataloaders['train']): 
				latent_feature = self.model(torch.autograd.Variable(data.cuda()), get_Latent = True)
				latent_feature = latent_feature.double().cpu().data.numpy()
				latent_feature = latent_feature.reshape((latent_feature.shape[0], latent_feature.shape[1]))
				label_var = np.array(label)
				if not len(x):
					x = latent_feature
					y = label_var
				else:
					x = np.concatenate((x, latent_feature))
					y = np.concatenate((y, label_var))
					assert len(x) == len(y), 'Data concatenation error'
				t.update()
				
		del latent_feature
		
		self.kmean.fit(x) 
		self.cluster_labels = self.__infer_cluster_labels__(y)
		
	def __predict__(self, X):
		latent_feature = self.model(torch.autograd.Variable(X.cuda()), get_Latent = True).double().cpu().data.numpy()
		latent_feature = latent_feature.reshape((latent_feature.shape[0], latent_feature.shape[1]))
		X_clusters = self.kmean.predict(latent_feature)
		return self.__infer_data_labels__(X_clusters)
		
	
	
	def validate(self, dataset = 'val'):
		y = []
		labels = []
		assert dataset in ['val', 'test'], 'dataset must be either "val" or "test" '
		print('testing')
		with tqdm(total=len(self.dataloaders[dataset])) as t:	
			for i, (data, label) in enumerate(self.dataloaders[dataset]): 
				output = self.__predict__(data)
				label_var = np.array(label)
				if not len(y):
					y = output
					labels = label_var
				else:
					y = np.concatenate((y, output))
					labels = np.concatenate((labels, label_var))
					assert len(labels) == len(y), 'Data concatenation error'
				t.update()
				
		acc = np.mean(y == labels)
		F1 = f1_score(	labels, y)
		print('Accuracy =', acc)
		print('F1 = ', F1)
		return acc, F1

	def train(self):
		start_epoch = 0
		best_val_loss = 0		

		self.__step__()
		self.__save_checkpoint__({
				'state_dict': self.model.state_dict(),
				'kmean': self.kmean,   #store the centroids to checkpoint
				'cluster_labels': self.cluster_labels
				}, 'Clustered')
		gc.collect()

		# evaluate on validation set
		self.validate()
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
		assert os.path.isfile(checkpointfile), 'model {} does not exist'.format(checkpointfile)
		
		logging.info("Loading checkpoint {}".format(checkpointfile))
		checkpoint = torch.load(checkpointfile)
		self.model.load_state_dict(checkpoint['state_dict'])
		self.model.eval()
		if checkpoint_type == 'Clustered':
			self.kmean = checkpoint['kmean']
			self.cluster_labels = checkpoint['cluster_labels']


	def __learning_rate_decay__(self, optimizer, decay_rate):
		if decay_rate < 1:
			for param_group in optimizer.param_groups:
				param_group['lr'] = param_group['lr'] * decay_rate
