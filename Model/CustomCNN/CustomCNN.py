from torch import nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
	def __init__(self, n_lf = 128): 
		super(BaseCNN, self).__init__()
		
		self.encoder = nn.Sequential(   #(4, 1, 1), 49.0), ((5, 2, 1), 24.0), ((4, 2, 0), 11.0), ((5, 2, 0), 4.0), ((4, 2, 0), 1.0)
			nn.Conv2d(3, 128, kernel_size = 4, stride=1, padding=1),  # b, 128, 49, 49
			nn.BatchNorm2d(num_features=128),
			nn.ReLU(True),
			nn.Conv2d(128, 256, kernel_size = 5, stride=2, padding=1),  # b, 256, 24, 24
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(True),
			nn.Conv2d(256, 512, kernel_size = 4, stride=2, padding=0),  # b, 512, 11, 11
			nn.BatchNorm2d(num_features=512),
			nn.ReLU(True),
			nn.Conv2d(512, 256, kernel_size = 5, stride=2, padding=0),  # b, 128, 4, 4
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(True),
			nn.Conv2d(256, n_lf, kernel_size = 4, stride=2, padding=0),  # b, 128, 1, 1
			nn.BatchNorm2d(num_features=n_lf),
		)
		self.decoder = nn.Sequential(
			nn.ReLU(True),
			nn.ConvTranspose2d(n_lf, 256, kernel_size = 4, stride=1, padding=1),  # b, 256, 2, 2
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(True),
			nn.ConvTranspose2d(256, 512, kernel_size = 3, stride=2, padding=1),  # b, 512, 3, 3
			nn.BatchNorm2d(num_features=512),
			nn.ReLU(True),
			nn.ConvTranspose2d(512, 256, kernel_size = 4, stride=2, padding=1),  # b, 1024, 6, 6
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(True),
			nn.ConvTranspose2d(256, 256, kernel_size = 4, stride=2, padding=1),  # b, 512, 12, 12
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(True),
			nn.ConvTranspose2d(256, 128, kernel_size = 5, stride=2, padding=1),  # b, 256, 25, 25
			nn.BatchNorm2d(num_features=128),
			nn.ReLU(True),
			nn.ConvTranspose2d(128, 3, kernel_size = 4, stride=2, padding=1),  # b, 3, 50, 50
			nn.Sigmoid(),
		)

	def forward(self, x, get_Latent = False):
		
		h = self.encoder(x)
		if not get_Latent:
			h = self.decoder(h)
			
		return h
        
