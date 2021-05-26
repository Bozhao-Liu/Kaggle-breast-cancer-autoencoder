import os
import torch
import logging
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

epslon = 1e-8

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __call__(self):
        return self.avg

def L2_loss(outputs, labels):
	return torch.sum(torch.pow(outputs - labels, 2))


def plot_AUC_SD(netlist, evalmatices):
	plt.clf()
	possitive_ratio = np.loadtxt("./data/possitive_ratio.txt", dtype=float)
	logging.warning('    Creating standard diviation image for {}'.format('-'.join(netlist)))
	png_file = 'Crossvalidation_Analysis_{}.PNG'.format('-'.join(netlist))

	if len(netlist) == 0:
		return


	plt.clf()
	fig, ax = plt.subplots(2)
	fig.suptitle('Accruacy, F1 for {}'.format('-'.join(netlist)))
	
	data = []
	for net in netlist:
		data.append(np.array(evalmatices[net]).T[0])

	ax[0].boxplot(data, showfliers=False)
	ax[0].set_ylabel('Accruacy')

	data = []
	for net in netlist:
		data.append(np.array(evalmatices[net]).T[1])

	ax[1].boxplot(data, showfliers=False)
	ax[1].set_ylabel('F1')
	ax[1].set_xticklabels(netlist, fontsize=10)

	logging.warning('    Saving standard diviation image for {} \n'.format('-'.join(netlist)))
	plt.savefig(png_file)

