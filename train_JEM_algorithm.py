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
use_tpu = params["use_tpu"]
save_every = params["save_every"]
save_path = params["save_path"]
save_model_prefix = params["save_path_prefix"]
save_buffer_prefix = params["save_buffer_path_prefix"]
image_prefix = params["image_prefix"]
load_from_checkpoint = params["load_from_checkpoint"]
print_every = params["print_every"]
start_epoch = params["start_epoch"]

if use_tpu != "False":
    # XLA SPECIFIC
    import torch_xla.core.xla_model as xm

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

### GPU / TPU ###
# define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if use_tpu != "False":
    # XLA SPECIFIC
    device = xm.xla_device()

print(f"device: {device}")

### REPLAY BUFFER ###
def create_random_buffer(size):
    return torch.FloatTensor(size, n_channels, im_size, im_size).uniform_(-1, 1)

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

    x_k, sample_indices = sample_buffer(buffer, batch_size, device)
    #x_k = Variable(init_samples, requires_grad=True)

    model.eval()
    for step in range(sgld_steps):
        x_k.requires_grad = True
        d_model_dx = torch.autograd.grad(model(x_k).sum(), x_k, retain_graph=True)[0] # TODO: remove retain graph=TRUE
        x_k = x_k.detach()
        x_k += sgld_step_size * d_model_dx + sgld_noise * torch.randn_like(x_k)
    model.train()

    sgld_samples = x_k.detach()

    #update replay buffer
    buffer[sample_indices] = sgld_samples.cpu()

    return sgld_samples

# define checkpointing and loading
def save_checkpoint(save_dir, epoch):
    print(f"saving model checkpoint at epoch {epoch} ...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), f"{save_dir}_{epoch}_epochs.pt")
    print("checkpoint saved!")

def save_model_and_buffer(save_dir, model, buffer, epoch, last=False):
    print(f"saving model and buffer checkpoint at epoch {epoch} ...")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.cpu() # TODO: this line doesn't seem to work when training with TPU
    checkpoint_dict = {
        "model": model.state_dict(),
        "buffer": buffer
    }

    if not last:
        torch.save(checkpoint_dict, f"{save_dir}ckpt_{epoch}_epochs.pt")

    else:
        torch.save(checkpoint_dict, f"{save_dir}last_ckpt.pt")
    model.to(device)
    print("model and buffer saved!")

def load_model_and_buffer(load_dir):
    print(f"loading model and buffer from {load_dir} ...")
    model = WRN_Energy(depth, widen_factor, 0.0, 10)
    checkpoint_dict = torch.load(load_dir)
    model.load_state_dict(checkpoint_dict["model"])
    model = model.to(device)
    buffer = checkpoint_dict["buffer"]

    return model, buffer

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

# define model and buffer
model = WRN_Energy(depth, widen_factor, 0.0, 10).to(device)
buffer = create_random_buffer(buffer_size)
if load_from_checkpoint != "False":
    model, buffer = load_model_and_buffer(load_from_checkpoint)

if not os.path.exists(image_prefix):
    os.makedirs(image_prefix)

# define the optimizer and criterion
supervised_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(.9, .999))

print(f"starting training from epoch {start_epoch} ...")

# setup the summary writer
writer = SummaryWriter(f'runs/JEM/')

experiment = replicate.init(
    path=".",
    params=params,
)

# set a learning rate schedule
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=decay_rate)

# define image saving functions
sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
plot = lambda path, x: torchvision.utils.save_image(torch.clamp(x, -1, 1), path, normalize=True, nrow=sqrt(x.size(0)))


# setup tensorboard logging steps
train_step = 0
val_step = 0

### TRAINING (Appendix A, Method 2, Algorithm 1) ###
for epoch in range(start_epoch, num_epochs):
    # Line 1: Sample (x, y)
    for i, (inputs, labels) in tqdm(enumerate(trainloader), total=len(trainset) // train_batch_size + 1):

        # obtain data
        inputs, labels = inputs.to(device), labels.to(device)

        loss = 0.

        if discriminative_weight > 0:
            # Line 2: xent(model(x), y)
            logits = model.classify(inputs)
            discriminative_loss = supervised_criterion(logits, labels)
            loss += discriminative_weight*discriminative_loss
            if i % print_every == 0:
                tqdm.write(f"disc_loss: {discriminative_loss} epoch: {epoch} it: {i}")
                writer.add_scalar("Loss/train", discriminative_loss, global_step=train_step)
                train_step += 1

        if generative_weight > 0:
            # Lines 4-7: Sample from buffer, run SGLD
            sgld_samples = run_sgld(model, buffer, train_batch_size, device)

            # Lines 8-9: add generative loss (I believe in the paper the signs on the two terms should be flipped)
            generative_loss = model(sgld_samples).mean() - model(inputs).mean()
            loss += generative_weight*generative_loss
            if i % print_every == 0:
                tqdm.write(f"gen_loss: {generative_loss} epoch: {epoch} it: {i}")
                writer.add_scalar("Loss/train_gen", generative_loss, global_step=train_step)
                train_step += 1

        if i % 100 == 0:
            plot_sgld_samples = run_sgld(model, buffer, train_batch_size, device)
            plot(os.path.join(image_prefix, f"sgld_{epoch}_{i}.png"), plot_sgld_samples)

        if loss.abs().item() > 1e8:
            print("Loss diverged! Restart training.")
            1 / 0

        # Line 10: back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if use_tpu != "False":
            # XLA SPECIFIC
            xm.mark_step()

    if epoch % save_every == 0:
        save_model_and_buffer(save_path, model, buffer, epoch, last=False)

        # Line 11: add to buffer, done in run_sgld function
    if epoch % eval_every == 0:

        corrects = []
        losses = []
        with torch.no_grad():
            model.eval()
            # check accuracy on validation set
            for test_inputs, test_labels in testloader:
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                logits = model.classify(test_inputs)
                loss = nn.CrossEntropyLoss(reduction='none')(logits, test_labels).cpu().numpy()
                losses.extend(loss)
                _, predicted = torch.max(logits.data, 1)
                correct = (predicted == test_labels).float().cpu().numpy()
                corrects.extend(correct)
            loss = np.mean(losses)
            acc = np.mean(corrects)
            tqdm.write(f"Epoch {epoch} validation accuracy: {acc}, Epoch validation loss: {loss}")
            save_model_and_buffer(save_path, model, buffer, epoch, last=True)

            writer.add_scalar("LR/lr", scheduler.get_last_lr()[0], global_step=val_step)
            writer.add_scalar("Loss/val", loss, global_step=val_step)
            writer.add_scalar("Loss/acc", acc, global_step=val_step)

            val_step += 1

        # save and log a checkpoint for replicate
            torch.save(model, "model.pth")
            experiment.checkpoint(
                path="model.pth",
                step=epoch,
                metrics={"accuracy": acc, "loss": loss},
                primary_metric=("accuracy", "maximize"),
                )
            model.train()

    scheduler.step()


























