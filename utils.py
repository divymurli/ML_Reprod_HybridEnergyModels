import torch
import os


def create_random_buffer(size, n_channels, im_size):

    """
    :param size: (int) number of images in buffer
    :param n_channels: (int) channels
    :param im_size: (int) image size
    :return: (arr) buffer
    """

    return torch.FloatTensor(size, n_channels, im_size, im_size).uniform_(-1, 1)


def run_sgld(model, x_k, sgld_steps, sgld_step_size, sgld_noise):

    """
    :param model: (obj) model
    :param x_k: (arr) sgld tensor
    :param sgld_steps: (int) number of sgld steps
    :param sgld_step_size: (float) sgld step size parameter
    :param sgld_noise: (float) sgld noise parameter
    :return:
    """

    model.eval()
    for step in range(sgld_steps):
        print(f"{step+1} of {sgld_steps} steps")
        x_k.requires_grad = True
        d_model_dx = torch.autograd.grad(model(x_k).sum(), x_k, retain_graph=True)[0] # TODO: remove retain graph=TRUE
        x_k = x_k.detach()
        x_k += sgld_step_size * d_model_dx + sgld_noise * torch.randn_like(x_k)

    sgld_samples = x_k.detach()

    return sgld_samples


def load_model_and_buffer(load_dir, model, device, with_energy=True):

    """
    :param load_dir: (str) directory from which to load model and buffer
    :param with_energy: (bool) if loading an ordinary model, or an energy-trained model with buffer
    :return: (obj) model, ((arr) buffer)
    """

    if with_energy:
        print(f"loading model and buffer from {load_dir} ...")
        checkpoint_dict = torch.load(load_dir)
        model.load_state_dict(checkpoint_dict["model"])
        model = model.to(device)
        buffer = checkpoint_dict["buffer"]

        return model, buffer

    else:
        print(f"loading model from {load_dir} ...")
        model_dict = torch.load(load_dir)
        model.load_state_dict(model_dict)
        model = model.to(device)

        return model


def save_checkpoint(model, save_dir, epoch, device):

    """
    :param model: (obj) model
    :param save_dir: (str) checkpoint save directory
    :param epoch: (int) checkpoint epoch
    :return: None
    """

    print(f"saving model checkpoint at epoch {epoch} ...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.cpu()  # TODO: this line doesn't seem to work when training with TPU
    torch.save(model.state_dict(), f"{save_path_prefix}_{epoch}_epochs.pt")
    model.to(device)
    print("checkpoint saved!")