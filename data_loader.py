import os

#from mnist.loader import MNIST
import numpy as np
import random
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from itertools import chain
import torchvision.transforms as transforms


class DatasetWrapper:
	class __DatasetWrapper:
		"""
		A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
		"""
		def __init__(self, cv_iters):
			"""
			create df for features and labels
			remove samples that are not shared between the two tables
			"""
			assert cv_iters > 2, 'Cross validation folds must be more than 2 folds'
			self.cv_iters = cv_iters
			datapath = './Data'
			self.dataset = [os.path.join(os.path.join(os.path.join(datapath, folder), label), image) 
						for folder in os.listdir(datapath)
							for label in os.listdir(os.path.join(datapath, folder)) 
								for image in os.listdir(os.path.join(os.path.join(datapath, folder), label))]
			self.Ind = np.arange(len(self.dataset))

			self.shuffle()

		def shuffle(self):
			"""
			categorize sample ID by label
			"""
			random.seed(231)
			random.shuffle(self.Ind)
			self.Ind = self.Ind[:int(len(self.Ind)/5)*5].reshape((self.cv_iters, -1))
			#index of valication set
			self.CVindex = 1
			self.Testindex = 0

		def next(self):
			'''
			rotate to the next cross validation process
			'''
			next_test = False
			if self.CVindex < self.cv_iters-1:
				self.CVindex += 1
				if self.Testindex < self.cv_iters-1:
					if self.Testindex == self.CVindex:
						self.CVindex += 1
				else:
					if self.Testindex == self.CVindex:
						self.CVindex = 0
						next_test = True
			else:
				self.CVindex = 0
				next_test = True
			
			if next_test:
				if self.Testindex < self.cv_iters-1:
					self.Testindex += 1
				else:
					self.Testindex = 0
					self.CVindex = 1

	instance = None
	def __init__(self, params, CViters, shuffle = 0):
		if not DatasetWrapper.instance:
			DatasetWrapper.instance = DatasetWrapper.__DatasetWrapper(params.CV_iters)

		if shuffle:
			DatasetWrapper.instance.shuffle()
		DatasetWrapper.Testindex = CViters[0]
		DatasetWrapper.CVindex = CViters[1]
			
			



	def __getattr__(self, name):
		return getattr(self.instance, name)

	def features(self, key):
		"""
		Args: 
			key:(string) value from dataset	
		Returns:
			features in list	
		"""
		return DatasetWrapper.instance.dataset[key]

	def label(self, key):
		"""
		Args: 
			key:(string) the sample key/id	
		Returns:
			label to number 8 or other
		"""
		return DatasetWrapper.instance.dataset[key]

	def next(self):
		DatasetWrapper.instance.next()

	def shuffle(self):
		DatasetWrapper.instance.shuffle()

	def __trainSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of trainning set
		"""

		ind = list(range(DatasetWrapper.instance.cv_iters))

		ind = np.delete(ind, [DatasetWrapper.instance.CVindex, DatasetWrapper.instance.Testindex])
		
		trainSet = DatasetWrapper.instance.Ind[ind].flatten()
		np.random.shuffle(trainSet)
		
		return trainSet
	
	def __valSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of validation set
		"""

		valSet = DatasetWrapper.instance.Ind[DatasetWrapper.instance.CVindex].flatten()
		np.random.shuffle(valSet)
		return valSet

	def __testSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of full dataset
		"""

		testSet = DatasetWrapper.instance.Ind[DatasetWrapper.instance.Testindex].flatten()
		np.random.shuffle(testSet)
		return testSet

	def getDataSet(self, dataSetType = 'train'):
		"""
		Args: 
			dataSetType: (string) 'train' or 'val'	
		Returns:
			dataset: (np.ndarray) array of key/id of data set
		"""

		if dataSetType == 'train':
			return self.__trainSet()

		if dataSetType == 'val':
			return self.__valSet()

		if dataSetType == 'test':
			return self.__testSet()

		return self.__testSet()
		


class imageDataset(Dataset):
	"""
	A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
	"""
	def __init__(self, dataSetType, params, CViters):
		"""
		initialize DatasetWrapper
		"""
		self.DatasetWrapper = DatasetWrapper(params, CViters)

		self.samples = self.DatasetWrapper.getDataSet(dataSetType)

		self.transformer = transforms.Compose([
					transforms.ToTensor(),
					transforms.Resize([250, 250])])

	def __len__(self):
		# return size of dataset
		return len(self.samples)



	def __getitem__(self, idx):
		"""
		Fetch feature and labels from dataset using index of the sample.

		Args:
		    idx: (int) index of the sample

		Returns:
		    feature: (Tensor) feature image
		    label: (int) corresponding label of sample
		"""
		sample = self.samples[idx]
		from PIL import Image
		image = Image.open(self.DatasetWrapper.features(sample))
		
		label = self.DatasetWrapper.label(sample)
		image = self.transformer(image)
		return image, label


def fetch_dataloader(types, params, CViters):
	"""
	Fetches the DataLoader object for each type in types.

	Args:
	types: (list) has one or more of 'train', 'val'depending on which data is required '' to get the full dataSet
	params: (Params) hyperparameters
	CViter: (tuple) positioning of (Testiter, CViter)

	Returns:
	data: (dict) contains the DataLoader object for each type in types
	"""
	dataloaders = {}
	assert CViters[0] != CViters[1], 'ERROR! Test set and validation set cannot be the same!'
	
	if len(types)>0:
		for split in types:
			if split in ['train', 'val', 'test']:
				dl = DataLoader(imageDataset(split, params, CViters), batch_size=params.batch_size, shuffle=True,
					num_workers=params.num_workers,
					pin_memory=params.cuda)

				dataloaders[split] = dl
	else:
		dl = DataLoader(imageDataset('', params, CViters), batch_size=params.batch_size, shuffle=True,
			num_workers=params.num_workers,
			pin_memory=params.cuda)

		return dl

	return dataloaders

