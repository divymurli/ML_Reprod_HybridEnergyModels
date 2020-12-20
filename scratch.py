import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from models import wide_resnet, resnet_official, wide_resnet_energy_output
import inception
import numpy as np

#This file is just a playground to mess around and make sure that things work. It's like scratch paper, and hence is not documented well.

#test to make sure the shapes go through
#net = wide_resnet.Wide_ResNet(28, 2, 0.0, 10)
#print(net)
#y = net(Variable(torch.randn(9, 3, 32, 32)))
#print(y)
#print(y.size())

#net = resnet_official.wrn_28_2()
#net = wide_resnet.WideResNet(28, 2, 0.0, 10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def load_model_and_buffer(load_dir, with_energy=True):

    if with_energy:
        print(f"loading model and buffer from {load_dir} ...")
        model = WRN_Energy(28, 2, 0.0, 10)
        checkpoint_dict = torch.load(load_dir)
        model.load_state_dict(checkpoint_dict["model"])
        model = model.to(device)
        buffer = checkpoint_dict["buffer"]

        return model, buffer
    else:
        print(f"loading model from {load_dir} ...")
        model = wide_resnet.WideResNet(depth, widen_factor, 0.0, 10)
        model_dict = torch.load(load_dir)
        model.load_state_dict(model_dict)
        model = model.to(device)

        return model

"""
#net = wide_resnet_energy_output.WideResNet_Penultimate(28, 10, 0.0, 10)
#print(net)
net = WRN_Energy(28, 10, 0.0, 10)
x = Variable(torch.randn(8, 3, 32, 32), requires_grad=True)
y = net(x)

y_prime = torch.autograd.grad(y.sum(), x, retain_graph=True)
print(y_prime[0].shape)

#print(net)
"""

## INCEPTION SCORE FROM BUFFER

plot = lambda path, x: torchvision.utils.save_image(torch.clamp(x, -1, 1), path, normalize=True, nrow=sqrt(x.size(0)))
sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))

# SINGLE BUFFER
"""
load_dir = "./artefacts/ckpt_145_epochs.pt"
img_save_path = "./artefacts/145_epochs_topk.png"
model, buffer = load_model_and_buffer(load_dir)

buffer = torch.clamp(buffer, -1, 1)
#predictions = inception.obtain_inception_predictions(buffer, cuda=False, batch_size=100, resize=True, save_preds=True)

predictions = np.load("inception_preds.npy")
top_k_preds, top_k_inds = inception.obtain_top_k(predictions, 64)

inception_score = inception.inception_score(top_k_preds, splits=1)
print(f"inception score for top k buffer preds: {inception_score}")
plot(img_save_path, buffer[top_k_inds])
"""

# ENSEMBLED BUFFERS
"""
load_dirs = [f"./artefacts/ckpt_{i}_epochs.pt" for i in range(125, 150, 5)]

buffers = []
for load_dir in load_dirs:
    _, buffer = load_model_and_buffer(load_dir)
    buffers.append(buffer)

ensembled_buffer = inception.ensemble_buffer(buffers)
predictions = inception.obtain_inception_predictions(ensembled_buffer, cuda=False, batch_size=100, resize=True, save_preds=True)
"""

"""
predictions = np.load("inception_preds_ensembled.npy")
top_k_preds, top_k_inds = inception.obtain_top_k(predictions, 10000)

inception_score = inception.inception_score(top_k_preds, splits=1)
print(f"inception score for top k buffer preds: {inception_score}")
"""

# LOADING FRESH SGLD GENERATIONS
steps = 100
load_dir = f"./fresh_sgld_samples/generated_samples_{steps}.pt"

samples = torch.load(load_dir)
print(samples["buffer"].shape)

samples = torch.clamp(samples["buffer"], -1, 1)

predictions = inception.obtain_inception_predictions(samples, cuda=False, batch_size=50, resize=True, save_preds=False)
inception_score = inception.inception_score(predictions, splits=1)
print(inception_score)





