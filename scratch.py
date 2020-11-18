import torch
import torch.nn as nn
from torch.autograd import Variable
from models import wide_resnet, resnet_official, wide_resnet_energy_output

#This file is just a playground to mess around and make sure that things work. It's like scratch paper, and hence is not documented well.

#test to make sure the shapes go through
#net = wide_resnet.Wide_ResNet(28, 2, 0.0, 10)
#print(net)
#y = net(Variable(torch.randn(9, 3, 32, 32)))
#print(y)
#print(y.size())

#net = resnet_official.wrn_28_2()
#net = wide_resnet.WideResNet(28, 2, 0.0, 10)

class WRN_Energy(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WRN_Energy, self).__init__()

        self.network = wide_resnet_energy_output.WideResNet_Penultimate(depth, widen_factor, dropout_rate, num_classes)
        self.classification_layer = nn.Linear(self.network.penultimate_layer, num_classes)

    def classify(self, x):
        penultimate_output = self.network(x)
        return self.classification_layer(penultimate_output).squeeze()

    def forward(self, x):
        logits = self.classify(x)

        energy = torch.logsumexp(logits, 1)

        return energy




#net = wide_resnet_energy_output.WideResNet_Penultimate(28, 10, 0.0, 10)
#print(net)
net = WRN_Energy(28, 10, 0.0, 10)
x = Variable(torch.randn(8, 3, 32, 32), requires_grad=True)
y = net(x)

y_prime = torch.autograd.grad(y.sum(), x, retain_graph=True)
print(y_prime[0].shape)

#print(net)