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
		return loadBaseCNN()
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

	logging.warning("No model with the name {} found, please check your spelling.".format(netname))
	logging.warning("Net List:")
	logging.warning("    basecnn")
	logging.warning("    basedbn")
	sys.exit()
    
def loadBaseCNN():
	from Model.BaseCNN.BaseCNN import BaseCNN
	logging.warning("Loading Base Model")
	return BaseCNN()


