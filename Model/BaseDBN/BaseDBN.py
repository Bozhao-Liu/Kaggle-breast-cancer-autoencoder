import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDBN(nn.Module):
	def __init__(self, hid_dim=10):
		super(BaseDBN,self).__init__()
		
		self.encoder=nn.Sequential(
			nn.Flatten(start_dim=1, end_dim=-1),
			nn.Linear(3 * 50 * 50, 2000),
			nn.BatchNorm1d(num_features = 2000),
			nn.ReLU(True),
			nn.Linear(2000, 1000),
			nn.BatchNorm1d(num_features = 1000),
			nn.ReLU(True),
			nn.Linear(1000, 500),
			nn.BatchNorm1d(num_features = 500),
			nn.ReLU(True),
			nn.Linear(500, hid_dim),
			nn.BatchNorm1d(num_features = hid_dim),
			nn.ReLU(True),
		)


		self.decoder=nn.Sequential(
			nn.Linear(hid_dim, 500),
			nn.BatchNorm1d(num_features = 500),
			nn.ReLU(True),
			nn.Linear(500, 1000),
			nn.BatchNorm1d(num_features = 1000),
			nn.ReLU(True),
			nn.Linear(1000, 2000),
			nn.BatchNorm1d(num_features = 2000),
			nn.ReLU(True),
			nn.Linear(2000, 3 * 50 * 50),
			nn.Sigmoid(),
		)



	def forward(self, x, get_Latent = False):
		h = self.encoder(x)
		if not get_Latent:
			h = self.decoder(h)
			h = h.view(x.shape)

		return h
