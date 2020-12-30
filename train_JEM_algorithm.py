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

from models import wide_resnet_energy_output
import utils

dir_path = os.path.dirname(os.path.realpath(__file__))
p = os.path.join(dir_path, 'params.json')
with open(p, 'r') as f:
    params = json.load(f)


if params["use_tpu"] != "False":
    # XLA SPECIFIC
    import torch_xla.core.xla_model as xm

# IMAGE CHARACTERISTICS ###
# define image parameters
n_channels = 3
im_size = 32


# DATA LOADING AND AUGMENTATION ###
# normalize all pixel values to be in [-1, 1] and add Gaussian noise with mean zero, variance gaussian_noise_var
# using the same train/test data augmentation as in the paper's code
transform_train = transforms.Compose(
            [transforms.Pad(4, padding_mode="reflect"),
             transforms.RandomCrop(im_size),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2488, 0.2453, 0.2633)),
             lambda x: x + params["gaussian_noise_var"] * torch.randn_like(x)]
)

transform_test = transforms.Compose(
            [transforms.ToTensor(),
             # normalize by the mean, stdev of the training set
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2488, 0.2453, 0.2633)),
             lambda x: x + params["gaussian_noise_var"] * torch.randn_like(x)]
)

# obtain data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=params["train_batch_size"],
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=params["test_batch_size"],
                                         shuffle=False, num_workers=2)

# GPU / TPU ###
# define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if params["use_tpu"] != "False":
    # XLA SPECIFIC
    device = xm.xla_device()

print(f"device: {device}")


# DEFINE MODEL ARCHITECTURE ###
model = wide_resnet_energy_output.WRN_Energy(params["depth"], params["widen_factor"], 0.0, 10).to(device)
buffer = utils.create_random_buffer(params["buffer_size"], n_channels, im_size)
if params["load_from_checkpoint"] != "False":
    model, buffer = utils.load_model_and_buffer(params["last_ckpt"], model, device)


# HELPER FUNCTIONS ###
#sample buffer initially
def sample_buffer(buffer, batch_size, device):

    """
    :param buffer: (arr) replay buffer
    :param batch_size: (int) batch size
    :param device: (obj) device
    :return: (arr) buffer samples, (arr) buffer sample indices
    """

    # sample from buffer
    buffer_length = buffer.size()[0]
    sample_indices = torch.randint(0, buffer_length, (batch_size, ))
    buffer_samples = buffer[sample_indices]

    # sample random
    random_samples = utils.create_random_buffer(batch_size, n_channels, im_size)
    reinit = (torch.rand(batch_size) < params["reinit_freq"]).float()[:, None, None, None]

    samples = (1-reinit) * buffer_samples + reinit * random_samples

    return samples.to(device), sample_indices


# run sgld
def train_sgld(model, buffer, batch_size, device):

    """
    :param model: (obj) model
    :param buffer: (arr) replay buffer
    :param batch_size: (int) batch size
    :param device: (obj) device
    :return: (arr) sgld samples
    :intermediate: update buffer
    """

    x_0, sample_indices = sample_buffer(buffer, batch_size, device)

    model.eval()
    x_k = utils.run_sgld(model, x_0, params["sgld_steps"], params["sgld_step_size"], params["sgld_noise"])
    model.train()

    sgld_samples = x_k.detach()

    #update replay buffer
    buffer[sample_indices] = sgld_samples.cpu()

    return sgld_samples


# TRAINING ###


def train(params):

    # path to drop generated images
    if not os.path.exists(params["image_prefix"]):
        os.makedirs(params["image_prefix"])

    # define the optimizer and criterion
    supervised_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], betas=(.9, .999))

    start_epoch = params["start_epoch"]
    print(f"starting training from epoch {start_epoch} ...")

    # setup the summary writer
    writer = SummaryWriter(f'runs/JEM/')

    experiment = replicate.init(
        path=".",
        params=params,
    )

    # set a learning rate schedule
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=params["decay_epochs"], gamma=params["decay_rate"])

    # define image saving functions
    sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
    plot = lambda path, x: torchvision.utils.save_image(torch.clamp(x, -1, 1), path, normalize=True, nrow=sqrt(x.size(0)))


    # setup tensorboard logging steps
    train_step = 0
    val_step = 0

    ### TRAINING (Appendix A, Method 2, Algorithm 1) ###
    for epoch in range(params["start_epoch"], params["num_epochs"]):
        # Line 1: Sample (x, y)
        for i, (inputs, labels) in tqdm(enumerate(trainloader), total=len(trainset) // params["train_batch_size"] + 1):

            # obtain data
            inputs, labels = inputs.to(device), labels.to(device)

            loss = 0.

            if params["discriminative_weight"] > 0:
                # Line 2: xent(model(x), y)
                logits = model.classify(inputs)
                discriminative_loss = supervised_criterion(logits, labels)
                loss += params["discriminative_weight"]*discriminative_loss
                if i % params["print_every"] == 0:
                    tqdm.write(f"disc_loss: {discriminative_loss} epoch: {epoch} it: {i}")
                    writer.add_scalar("Loss/train", discriminative_loss, global_step=train_step)
                    train_step += 1

            if params["generative_weight"] > 0:
                # Lines 4-7: Sample from buffer, run SGLD
                sgld_samples = train_sgld(model, buffer, params["train_batch_size"], device)

                # Lines 8-9: add generative loss (I believe in the paper the signs on the two terms should be flipped)
                generative_loss = model(sgld_samples).mean() - model(inputs).mean()
                loss += params["generative_weight"]*generative_loss
                if i % params["print_every"] == 0:
                    tqdm.write(f"gen_loss: {generative_loss} epoch: {epoch} it: {i}")
                    writer.add_scalar("Loss/train_gen", generative_loss, global_step=train_step)
                    train_step += 1

            if i % 100 == 0:
                plot_sgld_samples = train_sgld(model, buffer, params["train_batch_size"], device)
                plot(os.path.join(params["image_prefix"], f"sgld_{epoch}_{i}.png"), plot_sgld_samples)

            if loss.abs().item() > 1e8:
                print("Loss diverged! Restart training.")
                1 / 0

            # Line 10: back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if params["use_tpu"] != "False":
                # XLA SPECIFIC
                xm.mark_step()

        if epoch % params["save_every"] == 0:
            utils.save_model_and_buffer(params["save_path"], model, buffer, epoch, device, last=False)

            # Line 11: add to buffer, done in run_sgld function
        if epoch % params["eval_every"] == 0:

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
                utils.save_model_and_buffer(params["save_path"], model, buffer, epoch, device, last=True)

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


if __name__ == "__main__":

    train(params)























