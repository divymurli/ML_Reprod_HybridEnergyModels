import torch
from torch.autograd import Variable
from models import wide_resnet

#This file is just a playground to mess around and make sure that things work. It's like scratch paper, and hence is not documented well.

#test to make sure the shapes go through
net = wide_resnet.Wide_ResNet(28, 2, 0.0, 10)
y = net(Variable(torch.randn(9, 3, 32, 32)))
print(y)
print(y.size())