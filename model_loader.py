import os
import sys
import torch
import logging
import torch.nn as nn
def loadModel(params, netname = 'basecnn', dropout_rate = 0.5, channels = 1):
	Netpath = 'Model'

	Netfile = os.path.join(Netpath, netname)
	Netfile = os.path.join(Netfile, netname + '.py')
	assert os.path.isfile(Netfile), "No python file found for {}, (file name is case sensitive)".format(Netfile)
	netname = netname.lower()
	if netname == 'basecnn':
		return loadBaseCNN(params.num_LatentFeatures)
	if netname == 'basedbn':
		return loadBaseDBN(params.num_LatentFeatures)
	if netname == 'alexae':
		return loadAlexAE(params.num_LatentFeatures)
	if netname == 'resae':
		return loadResAE(params.num_LatentFeatures)
	if netname == 'customcnn':
		return loadCustomCNN(params.num_LatentFeatures)
	else:
		logging.warning("No model with the name {} found, please check your spelling.".format(netname))
		logging.warning("Net List:")
		logging.warning("    basecnn")
		logging.warning("    basedbn")
		sys.exit()

def get_model_list(netname = ''):
	
	if netname == '':
		return ['BaseCNN', 'BaseDBN']
		
	netname = netname.lower()
	if netname == 'basecnn':
		return ['BaseCNN']
	if netname == 'basedbn':
		return ['BaseDBN']
	if netname == 'alexae':
		return ['AlexAE']
	if netname == 'resae':
		return ['ResAE']
	if netname == 'customcnn':
		return ['CustomCNN']

	logging.warning("No model with the name {} found, please check your spelling.".format(netname))
	logging.warning("Net List:")
	logging.warning("    basecnn")
	logging.warning("    basedbn")
	logging.warning("    alexae")
	logging.warning("    customcnn")
	sys.exit()
    
def loadBaseCNN(num_LatentFeatures):
	from Model.BaseCNN.BaseCNN import BaseCNN
	logging.warning("Loading Base Model")
	return BaseCNN(num_LatentFeatures)

def loadBaseDBN(num_LatentFeatures):
	from Model.BaseDBN.BaseDBN import BaseDBN
	logging.warning("Loading Base Model")
	return BaseDBN(num_LatentFeatures)

def loadVAE(num_LatentFeatures):
	from Model.VAE.VAE import VAE
	logging.warning("Loading Base Model")
	return VAE(num_LatentFeatures)
	
def loadAlexAE(num_LatentFeatures):
	from Model.AlexAE.AlexAE import AlexAE
	logging.warning("Loading Base Model")
	return AlexAE(num_LatentFeatures)
	
def loadResAE(num_LatentFeatures):
	from Model.ResAE.ResAE import ResAE
	logging.warning("Loading Base Model")
	return ResAE(num_LatentFeatures)
	
def loadCustomCNN(num_LatentFeatures):
	from Model.CustomCNN.CustomCNN import BaseCNN
	logging.warning("Loading Base Model")
	return BaseCNN(num_LatentFeatures)
	

		
