import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from utils import load_state_dict_from_url



model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):

	def __init__(self, num_classes=1000):
		super(AlexNet, self).__init__()
		self.features = nn.Sequential(
		    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
		    nn.ReLU(inplace=True),
		    nn.MaxPool2d(kernel_size=3, stride=2),
		    nn.Conv2d(64, 192, kernel_size=5, padding=2),
		    nn.ReLU(inplace=True),
		    nn.MaxPool2d(kernel_size=3, stride=2),
		    nn.Conv2d(192, 384, kernel_size=3, padding=1),
		    nn.ReLU(inplace=True),
		    nn.Conv2d(384, 256, kernel_size=3, padding=1),
		    nn.ReLU(inplace=True),
		    nn.Conv2d(256, 256, kernel_size=3, padding=1),
		    nn.ReLU(inplace=True),
		    nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Sequential(
		    nn.Dropout(),
		    nn.Linear(256 * 6 * 6, 4096),
		    nn.ReLU(inplace=True),
		    nn.Dropout(),
		    nn.Linear(4096, 4096),
		    nn.ReLU(inplace=True),
		    nn.Linear(4096, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.classifier(x)
		return x


class RevertAlexNet(nn.Module):

	def __init__(self, num_classes=1000):
		super(RevertAlexNet, self).__init__()
		self.classifier = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(num_classes, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 256 * 6 * 6),
			)
		self.features = nn.Sequential(#(((5, 1, 1), 8), ((4, 2, 1), 16), ((4, 2, 2), 30), ((6, 2, 1), 62), ((6, 2, 1), 126), ((4, 2, 2), 250))492
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(256, 256, kernel_size=5, stride=1, padding=1),
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(256, 384, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(num_features=384),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=2),
			nn.BatchNorm2d(num_features=192),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(192, 192, kernel_size=6, stride=2, padding=1),
			nn.BatchNorm2d(num_features=192),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(192, 64, kernel_size=6, stride=2, padding=1),
			nn.BatchNorm2d(num_features=64),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=2),
			nn.Sigmoid(),
			)


	def forward(self, x):
		x = self.classifier(x)
		x = x.view(x.size(0), 256, 6, 6)
		x = self.features(x)
		
		return x
        
