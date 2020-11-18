import json
import os

import numpy as np
import replicate
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import wide_resnet_energy_output

dir_path = os.path.dirname(os.path.realpath(__file__))
p = os.path.join(dir_path, 'params.json')
with open(p, 'r') as f:
    params = json.load(f)

### HYPERPARAMETERS ###
gaussian_noise_var = params["gaussian_noise_var"]
model_type = params["model"]
depth = params["depth"]
widen_factor = params["widen_factor"]
dropout_rate = params["dropout_rate"]
num_classes = params["num_classes"]
train_batch_size = params["train_batch_size"]
test_batch_size = params["test_batch_size"]
learning_rate = params["learning_rate"]
num_epochs = params["num_epochs"]
decay_epochs = params["decay_epochs"]
decay_rate = params["decay_rate"]
eval_every = params["eval_every"]
reinit_freq = params["reinit_freq"]
sgld_step_size = params["sgld_step_size"]
sgld_noise = params["sgld_noise"]
discriminative_weight = params["discriminative_weight"]
generative_weight = params["generative_weight"]
sgld_steps = params["sgld_steps"]
buffer_size=params["buffer_size"]

### IMAGE CHARACTERISTICS ###
# define image parameters
n_channels = 3
im_size = 32

### DATA LOADING AND AUGMENTATION ###
# normalize all pixel values to be in [-1, 1] and add Gaussian noise with mean zero, variance gaussian_noise_var
# using the same train/test data augmentation as in the paper's code
transform_train = transforms.Compose(
            [transforms.Pad(4, padding_mode="reflect"),
             transforms.RandomCrop(im_size),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2488, 0.2453, 0.2633)),
             lambda x: x + gaussian_noise_var * torch.randn_like(x)]
)

transform_test = transforms.Compose(
            [transforms.ToTensor(),
             # normalize by the mean, stdev of the training set
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2488, 0.2453, 0.2633)),
             lambda x: x + gaussian_noise_var * torch.randn_like(x)]
)

# obtain data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=2)

### GPU ###
# define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



### REPLAY BUFFER ###
def create_random_buffer(size):
    return torch.FloatTensor(size, n_channels, im_size, im_size).uniform_(-1,1)

#sample buffer initially
def sample_buffer(buffer, batch_size, device):

    # sample from buffer
    buffer_length = buffer.size()[0]
    sample_indices = torch.randint(0, buffer_length, (batch_size, ))
    buffer_samples = buffer[sample_indices]

    # sample random
    random_samples = create_random_buffer(batch_size)
    reinit = (torch.rand(batch_size) < reinit_freq).float()[:, None, None, None]

    samples = (1-reinit) * buffer_samples + reinit * random_samples

    return samples.to(device), sample_indices

# run sgld
def run_sgld(model, buffer, batch_size, device):

    init_samples, sample_indices = sample_buffer(buffer, batch_size, device)

    x_k = Variable(init_samples, requires_grad=True)

    model.eval()
    for step in range(sgld_steps):
        print(f"step: {step}")
        d_model_dx = torch.autograd.grad(model(x_k).sum(), x_k, retain_graph=True)[0]
        x_k.data += sgld_step_size * d_model_dx + sgld_noise * torch.randn_like(x_k)
    model.train()

    sgld_samples = x_k.detach()

    #update replay buffer
    buffer[sample_indices] = sgld_samples.cpu()

    return sgld_samples

### DEFINING MODEL AND INITIALIZING BUFFER ###
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

#define model and buffer
model = WRN_Energy(28, 10, 0.0, 10)
buffer = create_random_buffer(buffer_size)

#debugging
print(buffer.shape)

init_samples, sample_indices = sample_buffer(buffer, train_batch_size, buffer)
sgld_samples = run_sgld(model, buffer, train_batch_size, device)























