import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from utils import load_state_dict_from_url

from Model.ResAE.ResNet import ResNet, ReverseResNet


__all__ = ['AlexNet', 'alexnet']


class ResAE(nn.Module):
	def __init__(self, n_lf = 1000): 
		super(ResAE, self).__init__()
		self.encoder = ResNet(num_classes = n_lf)
		self.decoder = ReverseResNet(num_classes = n_lf)
		
	def forward(self, x, get_Latent = False):
		h = self.encoder(x)
		if not get_Latent:
			h = self.decoder(h)
			assert h.shape == x.shape, "Decoder error, OutShape = {}, Inshape = {}".format(h.shape, x.shape)			
		return h
        
