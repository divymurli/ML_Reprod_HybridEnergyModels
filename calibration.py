import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import os
import json

from models import wide_resnet, wide_resnet_energy_output

dir_path = os.path.dirname(os.path.realpath(__file__))
p = os.path.join(dir_path, 'params.json')
with open(p, 'r') as f:
    params = json.load(f)

gaussian_noise_var = params["gaussian_noise_var"]
depth = params["depth"]
widen_factor = params["widen_factor"]
test_batch_size = params["test_batch_size"]

### IMAGE CHARACTERISTICS ###
# define image parameters
n_channels = 3
im_size = 32

### DATA LOADING AND AUGMENTATION ###
# normalize all pixel values to be in [-1, 1] and add Gaussian noise with mean zero, variance gaussian_noise_var
# using the same train/test data augmentation as in the paper's code

transform_test = transforms.Compose(
            [transforms.ToTensor(),
             # normalize by the mean, stdev of the training set
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2488, 0.2453, 0.2633)),
             lambda x: x + gaussian_noise_var * torch.randn_like(x)]
)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=2)

#define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
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

#Helper functions

def load_model_and_buffer(load_dir, with_energy=True):

    if with_energy:
        print(f"loading model and buffer from {load_dir} ...")
        model = WRN_Energy(depth, widen_factor, 0.0, 10)
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

def correct_and_confidence(model, loader, with_energy=True):
    with torch.no_grad():
        model.eval()
        confidences = []
        corrects = []
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if with_energy:
                logits = model.classify(inputs)
            else:
                logits = model(inputs)

            _, predicted = torch.max(logits.data, 1)
            softmaxed_logits = nn.Softmax(dim=1)(logits)
            confidence, _ = torch.max(softmaxed_logits.data, 1)
            confidences.extend(confidence)
            correct = (predicted == labels).float().cpu().numpy()
            corrects.extend(correct)

    return np.array(sorted(list(zip(corrects, confidences)), key=lambda x: x[1]))

def calibration_buckets(zipped_corr_conf):

    thresholds = np.linspace(0, 1, 21)
    corrects = zipped_corr_conf[:, 0]
    confidences = zipped_corr_conf[:, 1]

    buckets = [(thresholds[i], thresholds[i+1]) for i in range(len(thresholds) - 1)]
    bucket_accs = []

    for bucket in buckets:
        total = 0
        correct = 0
        for i in range(len(confidences)):
            if confidences[i] > bucket[0] and confidences[i] < bucket[1]:
                total += 1
                correct += corrects[i]
        if total != 0:
            bucket_acc = correct / total
        else:
            bucket_acc = 0.
        bucket_accs.append(bucket_acc)

    return buckets, bucket_accs


#load_dir = "./artefacts/ckpt_145_epochs.pt"
load_dir = "./artefacts/model_145_epochs.pt"

# Analysis

"""
model, buffer = load_model_and_buffer(load_dir)

zipped_corr_conf = correct_and_confidence(model, testloader)

with open("zipped_corr_conf.npy", "wb") as f:
    np.save(f, zipped_corr_conf)
"""

"""
model = load_model_and_buffer(load_dir, with_energy=False)

zipped_corr_conf = correct_and_confidence(model, testloader, with_energy=False)

with open("zipped_corr_conf_supervised.npy", "wb") as f:
    np.save(f, zipped_corr_conf)
"""

zipped_corr_conf = np.load("zipped_corr_conf_supervised.npy")
buckets, bucket_accs = calibration_buckets(zipped_corr_conf)


fig = plt.figure(figsize=(10,10), facecolor="white")
ax = plt.subplot(111)
ax.bar(np.arange(20), height=bucket_accs)
ax.set_xticks(np.arange(20))
x = np.linspace(*ax.get_xlim())
y = np.linspace(*ax.get_ylim())
ax.plot(x, y, linestyle='dashed', color='red')
ax.set_xlabel("bucket")
ax.set_ylabel("bucket accuracy")

plt.show()

print(buckets)
print(bucket_accs)
print(np.linspace(0, 1, 21)[1:])


