import csv
import os
from PIL import Image
import numpy as np
import random
from random import shuffle
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F

class ImageFolder(data.Dataset):
	def __init__(self, root, image_size=224, mode='train'):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		self.GT_paths = root[:-1]+'_GT/'
		GT_file = csv.DictReader(open(self.GT_paths+mode+'_label.csv', encoding='UTF-8-sig'))
		self.GT_list = [i for i in GT_file]
		#self.GT_list = []
		#with open(self.GT_paths+mode+'_label.csv', 'r') as fi:
		#	for i in fi:
		#		self.GT_list.append(i)
		self.image_size = image_size
		self.mode = mode
		print("image count in {} path :{}".format(self.mode,len(self.GT_list)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.GT_list[index]['FileName']
		#image_path = self.GT_list[index].split(' ')[0]
		image = Image.open(self.root+image_path)
		image_channels = len(image.split())
		if image_channels == 4:
			image = Image.open(self.root+image_path).convert("RGB")
		GT = int(self.GT_list[index]['Code'])-1
		#GT = int(self.GT_list[index].split(' ')[1])-1
		Transform = []
		if self.mode == 'train':
			Transform.append(T.Resize((256, 256)))
			Transform.append(T.RandomResizedCrop(224))
			Transform.append(T.RandomHorizontalFlip())
		elif self.mode == 'test' or self.mode == 'valid':
			Transform.append(T.Resize((256,256)))
			Transform.append(T.CenterCrop(224))
		Transform.append(T.ToTensor())
		Transform = T.Compose(Transform)
		image = Transform(image)
		Norm_ = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
		image = Norm_(image)
		return image, GT, image_path

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.GT_list)

def get_loader(image_path, image_size, batch_size, shuffle=True, num_workers=1, mode='train'):
	"""Builds and returns Dataloader."""
	dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=shuffle,
								  num_workers=num_workers)
	return data_loader
