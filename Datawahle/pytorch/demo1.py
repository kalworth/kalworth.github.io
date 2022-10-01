import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import datasets
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 256  # 批次补充
num_workers = 0   # 对于Windows用户，这里应设置为0，否则会出现多线程错误
lr = 1e-4
epochs = 20

image_size = 28
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.ToTensor()
])

train_data = datasets.FashionMNIST(
    root='./', train=True, download=True, transform=data_transform)
test_data = datasets.FashionMNIST(
    root='./', train=False, download=True, transform=data_transform)
