import json
import os

import numpy as np
import replicate
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import wide_resnet, resnet_official

# XLA SPECIFIC
import torch_xla.core.xla_model as xm

dir_path = os.path.dirname(os.path.realpath(__file__))
p = os.path.join(dir_path, 'params.json')
with open(p, 'r') as f:
    params = json.load(f)

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
im_size = 32
n_channels = 3

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

# define the device
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# XLA SPECIFIC
device = xm.xla_device()

# define the model
#
if model_type == "resnet_official":
    model = resnet_official.wrn_28_2().to(device)
else:
    model = wide_resnet.WideResNet(depth=depth, widen_factor=widen_factor, dropout_rate=dropout_rate,
                                    num_classes=num_classes).to(device)
# define the optimizer and criterion
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(.9, .999))

# setup the summary writer
writer = SummaryWriter(f'runs/wideresnet/')

print("starting training ...")

experiment = replicate.init(
    path=".",
    params=params,
)

# set a learning rate schedule
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=decay_rate)

train_step = 0
val_step = 0
# create a real training loop
for epoch in range(num_epochs):
    for i, (inputs, labels) in tqdm(enumerate(trainloader), total=len(trainset) // train_batch_size + 1):

        # obtain data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero gradients
        optimizer.zero_grad()

        # forward, backward, loss
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        # XLA SPECIFIC
        xm.mark_step()

        if i % 5 == 0:
            tqdm.write(f"loss: {loss.item()}")
            writer.add_scalar("Loss/train", loss, global_step=train_step)
            train_step +=1

    if epoch % eval_every == 0:

        corrects = []
        losses = []
        with torch.no_grad():
            model.eval()
            # check accuracy on validation set
            for test_inputs, test_labels in testloader:
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                logits = model(test_inputs)
                loss = nn.CrossEntropyLoss(reduction='none')(logits, test_labels).cpu().numpy()
                losses.extend(loss)
                _, predicted = torch.max(logits.data, 1)
                correct = (predicted == test_labels).float().cpu().numpy()
                corrects.extend(correct)
            loss = np.mean(losses)
            acc = np.mean(corrects)
            tqdm.write(f"Epoch {epoch} validation accuracy: {acc}, Epoch validation loss: {loss}")
            writer.add_scalar("LR/lr", scheduler.get_last_lr()[0], global_step=val_step)
            writer.add_scalar("Loss/val", loss, global_step=val_step)
            writer.add_scalar("Loss/acc", acc, global_step=val_step)

            val_step+=1

            # save and log a checkpoint
            torch.save(model, "model.pth")
            experiment.checkpoint(
                path="model.pth",
                step=epoch,
                metrics={"accuracy": acc, "loss": loss},
                primary_metric=("accuracy", "maximize"),
            )

            model.train()

    scheduler.step()
