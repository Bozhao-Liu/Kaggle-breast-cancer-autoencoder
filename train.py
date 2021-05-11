import argparse

import torch
import logging
import os

from itertools import permutations 
from Solver import Solver
import torch.backends.cudnn as cudnn
from model_loader import get_model_list
from utils import set_logger, set_params

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
parser = argparse.ArgumentParser(description='PyTorch Deep Neural Net Training')
parser.add_argument('--train', default = False, type=str2bool, 
			help="specify whether train the model or not (default: False)")
parser.add_argument('--model_dir', default='/media/bozhao/Data/Shared/Bozhao/NTNU/BreastCancer/Model', 
			help="Directory containing params.json")
parser.add_argument('--resume', default = True, type=str2bool, 
			help='resume from latest checkpoint (default: True)')
parser.add_argument('--network', type=str, default = 'BaseCNN',
			help='select network to train on. leave it blank means train on all model')
parser.add_argument('--log', default='warning', type=str,
			help='set logging level')
parser.add_argument('--lrDecay', default=1.0, type=float,
			help='learning rate decay rate')



def main():
	assert torch.cuda.is_available(), "ERROR! GPU is not available."
	cudnn.benchmark = True
	args = parser.parse_args()
	netlist = get_model_list(args.network)
	for network in netlist:
		args.network = network
		set_logger('./Model', args.network, args.log)
		params = set_params('./Model', network)
		
		CV_iters = list(permutations(list(range(params.CV_iters)), 2))

		for i, CViter in enumerate(CV_iters):
			logging.warning('Cross Validation on iteration {}/{}'.format(i, len(CV_iters)))
			
			solver = Solver(args, params, CViter)
			
			if args.train:
				solver.train()


		
if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"]="0"
	main()
