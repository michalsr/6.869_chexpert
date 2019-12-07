import torch.nn as nn
import torchvision

def densenet121(num_classes, pretrained=False, **kwargs):
	model = torchvision.models.densenet121(pretrained=pretrained,**kwargs)
	num_features = model.classifier.in_features
	model.classifier = nn.Sequential(
		nn.Linear(num_features, num_features),
		nn.Dropout(p=.9),
		nn.Linear(num_features, num_classes),
		nn.Sigmoid()
	)
	return model

