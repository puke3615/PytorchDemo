# coding=utf-8
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, self.__flat(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __flat(self, x):
        shape = x.size()[1:]
        units = 1
        for layer in shape:
            units *= layer
        return units


net = Net()
print net

parameters = list(net.parameters())
for i, parameter in enumerate(parameters):
    size = parameter.size()
    print '第%d层结构 %s' % (i + 1, str(list(size)))
    units_num = 1
    for unit in size:
        units_num *= unit
    print '共有%d个参数' % units_num
