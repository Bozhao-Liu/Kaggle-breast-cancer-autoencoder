import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from utils import load_state_dict_from_url

from Model.AlexAE.AlexNet import AlexNet, RevertAlexNet


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexAE(nn.Module):
	def __init__(self, n_lf = 1000): 
		super(AlexAE, self).__init__()
		self.encoder = AlexNet()
		state_dict = load_state_dict_from_url(model_urls['alexnet'], progress = True)
		self.encoder.load_state_dict(state_dict)
		self.encoder.classifier[-1] = nn.Linear(4096, n_lf)
		self.decoder = RevertAlexNet(num_classes = n_lf)
		
	def forward(self, x, get_Latent = False):
		
		h = self.encoder(x)
		if not get_Latent:
			h = self.decoder(h)
			assert h.shape == x.shape, "Decoder error, OutShape = {}, Inshape = {}".format(h.shape, x.shape)			
		return h
        
