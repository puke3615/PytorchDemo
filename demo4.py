# coding=utf-8
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 加载数据
trainset = torchvision.datasets.CIFAR10('./data', download=False)
print len(trainset)

# 数据分割、乱序
trainloader = DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
