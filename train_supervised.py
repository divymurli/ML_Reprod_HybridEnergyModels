import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import json
import os
import replicate
from models import wide_resnet, resnet_official

dir_path = os.path.dirname(os.path.realpath(__file__))
p = os.path.join(dir_path, 'params.json')
with open(p, 'r') as f:
    params = json.load(f)
    f.close()

gaussian_noise_var = params["gaussian_noise_var"]
depth = params["depth"]
widen_factor = params["widen_factor"]
dropout_rate = params["dropout_rate"]
num_classes = params["num_classes"]
train_batch_size = params["train_batch_size"]
test_batch_size = params["test_batch_size"]
learning_rate = params["learning_rate"]
num_epochs = params["num_epochs"]
eval_every = params["eval_every"]

# normalize all pixel values to be in [-1, 1] and add Gaussian noise with mean zero, variance gaussian_noise_var
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     lambda x: x + gaussian_noise_var * torch.randn_like(x)]
)

# obtain data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=2)

#define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define the model
#model = wide_resnet.Wide_ResNet(depth=depth, widen_factor=widen_factor, dropout_rate=dropout_rate, num_classes=num_classes)
model = resnet_official.wrn_28_2().to(device)

#define the optimizer and criterion
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=[.9, .999])


print("starting training ...")

experiment = replicate.init(
        path=".",
        params={"learning_rate": params["learning_rate"], "num_epochs": params["num_epochs"]},
    )

#create a real training loop
for epoch in range(num_epochs):
    for i, data in tqdm(enumerate(trainloader), total=len(trainset)):

        #obtain data
        inputs, labels = data[0].to(device), data[1].to(device)

        #zero gradients
        optimizer.zero_grad()

        #forward, backward, loss
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            tqdm.write(f"loss: {loss.item()}")

    if epoch % eval_every == 0:
        #if i % 10 == 0:
            #total = 0
            corrects = []
            losses = []
        with torch.no_grad():
            model.eval()
            #check accuracy on validation set
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                logits = model(images)
                loss = nn.CrossEntropyLoss(reduction='none')(logits, labels).cpu().numpy()
                losses.extend(loss)
                _, predicted = torch.max(logits.data, 1)
                correct = (predicted == labels).float().cpu().numpy()
                corrects.extend(correct)
            loss = np.mean(losses)
            acc = np.mean(corrects)
            tqdm.write(f"Epoch {epoch} validation accuracy: {acc}, Epoch validation loss: {loss}")

            #save and log a checkpoint
            torch.save(model, "model.pth")
            experiment.checkpoint(
                path="model.pth",
                step=epoch,
                metrics={"accuracy": acc, "loss": loss},
                primary_metric=("accuracy", "maximize"),
            )

            model.train()