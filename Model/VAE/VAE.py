import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
	def __init__(self,input_dim=784,inter_dims=[500,500,2000],hid_dim=10):
		super(VAE,self).__init__()

		self.encoder=nn.Sequential(
			nn.Flatten(start_dim=1, end_dim=-1)
			nn.Linear(input_dim,1000),
			nn.BatchNorm1d(num_features=1000),
			nn.ReLU(True),
			nn.Linear(1000,1000),
			nn.BatchNorm1d(num_features=500),
			nn.ReLU(True),
			nn.Linear(500,2000),
			nn.BatchNorm1d(num_features=2000),
			nn.ReLU(True),
		)

		self.mu=nn.Linear(2000,hid_dim)
		self.SD=nn.Linear(2000,hid_dim)

		self.decoder=nn.Sequential(
			nn.Linear(hid_dim,2000),
			nn.BatchNorm1d(num_features=2000),
			nn.ReLU(True),
			nn.Linear(2000,500),
			nn.BatchNorm1d(num_features=500),
			nn.ReLU(True),
			nn.Linear(500,500),
			nn.BatchNorm1d(num_features=500),
			nn.ReLU(True),
			nn.Linear(500,input_dim),
			nn.Sigmoid(),
		)



	def forward(self, x, get_Latent = False):
			
		h = self.encoder(x)
		mu = self.mu(h)
		SD = self.SD(h)
		if get_Latent:
			h = torch.cat((mu, SD), dim = 1)
		else:
			h = torch.randn_like(mu)*torch.exp(SD/2) + mu
			h = self.decoder(h)
			h = h.view(x.shape)

		return h
