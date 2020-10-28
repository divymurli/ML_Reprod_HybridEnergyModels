import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from models import wide_resnet, resnet_official

# hyperparameters (eventually will be made user passable with argparse):
gaussian_noise_var = 3e-2
depth = 28
widen_factor = 2
dropout_rate = 0.0
num_classes = 10
train_batch_size = 32
test_batch_size = 32
learning_rate = 1e-3
num_epochs = 1

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

# for debugging purposes, define a truncated training set
#toy_train_set = list(enumerate(trainloader, 0))[:2]

# define the model
#model = wide_resnet.Wide_ResNet(depth=depth, widen_factor=widen_factor, dropout_rate=dropout_rate, num_classes=num_classes)
model = resnet_official.wrn_28_2()

#define the optimizer and criterion
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=[.9, .999])


# define a truncated training set just to make sure everything works as expected
#truncated_training_set = list(enumerate(trainloader, 0))[:2]

print("starting training ...")

#create a real training loop
for epoch in range(num_epochs):
    for i, data in tqdm(enumerate(trainloader), total=len(trainset)):


        #obtain data
        inputs, labels = data

        #zero gradients
        optimizer.zero_grad()

        #forward, backward, loss
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            tqdm.write(f"loss: {loss}")
        if i % 10 == 0:
            total = 0
            correct = 0
            model.eval()
            with torch.no_grad():
                #check accuracy on validation set
                for data in testloader:
                    images, labels = data
                    logits = model(images)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            tqdm.write(f"Accuracy of the network on 10000 test images: {correct}/{total}")
            model.train()