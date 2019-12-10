import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class ChestDataSet(Dataset):
	def __init__(self, root, image_list_file, transform=None,mode='train'):
		'''
		Args:
			root: root path to image directory.
			image_list_file: path to the file containing images 
				with corresponding labels.
			transform: optimal transform to be applied on a sample.
		'''
		imgs_path = []
		labels = []
		with open(image_list_file,"r") as f:
			for line in f:
				items=line.split()
				img_path=os.path.join(root,items[0])
				label = items[1:]
				
				imgs_path.append(img_path)
				labels.append(label)
		self.imgs_path = imgs_path
		self.labels = labels
		if transform is None:
			normalize = transforms.Normalize([.485,.456,.406],
							[0.229,.224,.225])
			if mode == 'train':
				transform = transforms.Compose([
					transforms.Resize(1024),
					transforms.CenterCrop(896),
					# transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					normalize,
				])
			self.transform = transform
		
		
	def __getitem__(self,index):
		'''
		Args:
			index: index of item
		Returns:
			image and its label
		'''
		img_path = self.imgs_path[index]
		img = Image.open(img_path).convert('RGB')
		label = torch.FloatTensor(list(map(float,self.labels[index])))
		if self.transform is not None:
			img = self.transform(img)
		return img,label
	def __len__(self):
		return len(self.imgs_path)
