from torch import nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
	def __init__(self):
		super(BaseCNN, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 128, kernel_size = 2, stride=2, padding=3),  # b, 128, 128, 128
			nn.BatchNorm2d(num_features=128),
			nn.ReLU(True),
			nn.MaxPool2d(2, stride=2),  # b, 128, 64, 64
			nn.Conv2d(128, 128, kernel_size = 4, stride=2, padding=1),  # b, 128, 32, 32
			nn.BatchNorm2d(num_features=128),
			nn.ReLU(True),
			nn.MaxPool2d(2, stride=2),  # b, 32, 16, 16
			nn.Conv2d(128, 64, kernel_size = 4, stride=2, padding=1),  # b, 64, 8, 8
			nn.BatchNorm2d(num_features=64),
			nn.ReLU(True),
			nn.MaxPool2d(2, stride=2),  # b, 64, 4, 4
			nn.Conv2d(64, 32, kernel_size = 4, stride=2, padding=1),  # b, 32, 2, 2
			nn.BatchNorm2d(num_features=32),
			nn.ReLU(True),
			nn.MaxPool2d(2, stride=2),  # b, 64, 1, 1
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(32, 64, kernel_size = 6, stride=1, padding=1),  # b, 64, 4, 4
			nn.BatchNorm2d(num_features=64),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 128, kernel_size = 4, stride=2, padding=1),  # b, 32, 8, 8
			nn.BatchNorm2d(num_features=128),
			nn.ReLU(True),
			nn.ConvTranspose2d(128, 128, kernel_size = 5, stride=3, padding=2),  # b, 32, 22, 22
			nn.BatchNorm2d(num_features=128),
			nn.ReLU(True),
			nn.ConvTranspose2d(128, 128, kernel_size = 5, stride=3, padding=2),  # b, 32, 64, 64
			nn.BatchNorm2d(num_features=128),
			nn.ReLU(True),
			nn.ConvTranspose2d(128, 128, kernel_size = 5, stride=2, padding=3),  # b, 32, 125, 125
			nn.BatchNorm2d(num_features=128),
			nn.ReLU(True),
			nn.ConvTranspose2d(128, 3, kernel_size = 4, stride=2, padding=1),  # b, 32, 250, 250
			nn.Sigmoid(),
		)

	def forward(self, x, get_Latent = False):
		
		if get_Latent:
			return self.encoder(x)
		else:
			h = self.encoder(x)
			h = self.decoder(h)
			return h
        
