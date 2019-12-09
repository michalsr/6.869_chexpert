from data.dataset import ChestDataSet
from config import opt
from torch.utils.data import DataLoader
import numpy as np
import torch

train_data = ChestDataSet(opt.data_root, opt.train_data_list, mode='train')
train_dataloader = DataLoader(train_data, 1, shuffle=True)

val_data = ChestDataSet(opt.data_root, opt.valid_data_list, mode = 'train')
val_dataloader = DataLoader(val_data, 1, shuffle = False)

test_data = ChestDataSet(opt.data_root, opt.test_data_list,mode='train')
test_dataloader = DataLoader(test_data, 1, shuffle=False)

bar = enumerate(test_dataloader)
classCnt = np.zeros(6).reshape(1, 6)
for i, (data, label) in bar:
    classCnt += label.numpy()
    pass
print(i, classCnt, classCnt/i)

bar = enumerate(val_dataloader)
classCnt = np.zeros(6).reshape(1, 6)
for i, (data, label) in bar:
    classCnt += label.numpy()
    pass
print(i, classCnt, classCnt/i)

bar = enumerate(train_dataloader)
classCnt = np.zeros(6).reshape(1, 6)
for i, (data, label) in bar:
    classCnt += label.numpy()
    pass
print(i, classCnt, classCnt/i)
