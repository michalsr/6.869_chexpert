import torch
import torch.utils.data
import torchvision
from data.dataset import ChestDataSet
import numpy as np
from config import opt

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = np.zeros(len(opt.classes))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            for clss, i in label, list(range(len(label))):
                label_to_count[i] += clss
                
        # weight for each sample
        weights = np.zeros(len(dataset))

        for idx in self.indices:
            c = 0
            label = self._get_label(dataset, idx)
            for clss, i in label, list(range(len(label))):
                if clss:
                    weights[idx] += (1/label_to_count[i])
                    c += 1
            weights[idx] /= c 

        # weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
        #            for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        img, label = dataset.__getitem__(idx)
        return label
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples