import json
import os

import numpy as np
import replicate
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models import wide_resnet, resnet_official, wide_resnet_energy_output

# Model
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

def run_fresh_sgld(model, x_k, sgld_steps, sgld_step_size, sgld_noise):

    model.eval()
    for step in range(sgld_steps):
        print(f"{step} of {sgld_steps} steps")
        x_k.requires_grad = True
        d_model_dx = torch.autograd.grad(model(x_k).sum(), x_k, retain_graph=True)[0] # TODO: remove retain graph=TRUE
        x_k = x_k.detach()
        x_k += sgld_step_size * d_model_dx + sgld_noise * torch.randn_like(x_k)

    sgld_samples = x_k.detach()

    return sgld_samples

def create_random_buffer(size, n_channels, im_size):
    return torch.FloatTensor(size, n_channels, im_size, im_size).uniform_(-1, 1)

def load_model_and_buffer(load_dir):
    print(f"loading model and buffer from {load_dir} ...")
    model = WRN_Energy(depth, widen_factor, 0.0, 10)
    checkpoint_dict = torch.load(load_dir)
    model.load_state_dict(checkpoint_dict["model"])
    model = model.to(device)
    buffer = checkpoint_dict["buffer"]

    return model, buffer

sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
plot = lambda path, x: torchvision.utils.save_image(torch.clamp(x, -1, 1), path, normalize=True, nrow=sqrt(x.size(0)))

dir_path = os.path.dirname(os.path.realpath(__file__))
p = os.path.join(dir_path, 'params.json')
with open(p, 'r') as f:
    params = json.load(f)

sgld_noise = params["sgld_noise"]
sgld_step_size = params["sgld_step_size"]
model_load_dir = "./artefacts/ckpt_145_epochs.pt"
depth = params["depth"]
widen_factor = params["widen_factor"]
sgld_steps = 50
save_dir = "./fresh_sgld_samples/"

# create the save directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

x_k = create_random_buffer(100, 3, 32)
plot(f"{save_dir}initial_start.png", x_k)
x_k = x_k.to(device)

model, _ = load_model_and_buffer(model_load_dir)
model = model.to(device)

sgld_samples = run_fresh_sgld(model, x_k, sgld_steps, sgld_step_size, sgld_noise)

plot(f"{save_dir}fresh_sgld_{sgld_steps}.png", sgld_samples)

sgld_cpu_samples = {"buffer": sgld_samples.cpu()}
torch.save(sgld_cpu_samples, f"{save_dir}generated_samples_{sgld_steps}.pt")




