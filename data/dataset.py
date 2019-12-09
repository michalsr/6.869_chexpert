import os
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import PIL
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
			if mode == 'train2':
				transform = transforms.Compose([
					transforms.Resize(256),
					transforms.CenterCrop(224),
					 #transforms.RandomHorizontalFlip(),
					#transforms.RandomRotate(270),
					transforms.ToTensor(),
					normalize,
				])
			if mode == 'train':
				transform = transforms.Compose([
				ImgAugTransform(),
				lambda x: PIL.Image.fromarray(x),
				transforms.RandomVerticalFlip(),
				transforms.ToTensor(),
				normalize
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
class ImgAugTransform:
  def __init__(self):
    self.aug = iaa.Sequential([
        iaa.Resize((224, 224)),
	iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
        #iaa.AddToHueAndSaturation(value=(-20, 20), per_channel=True),
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-20, 20), mode='symmetric')
    ])
      
  def __call__(self, img):
    img = np.array(img)
    return self.aug.augment_image(img)
