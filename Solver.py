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

def Solver(args, params, CViter, train_cluster= False, test = False):
	if test or train_cluster:
		from Sovler_Cluster import Sovler
		return Sovler(args, params, CViter, test)
	else:
		from Sovler_Autoencoder import Sovler
		return Sovler(args, params, CViter)

	
