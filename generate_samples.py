import json
import os
import torch

import torchvision
from models import wide_resnet_energy_output

dir_path = os.path.dirname(os.path.realpath(__file__))
p = os.path.join(dir_path, 'params.json')
with open(p, 'r') as f:
    params = json.load(f)


def run_fresh_sgld(model, x_k, sgld_steps, sgld_step_size, sgld_noise):

    model.eval()
    for step in range(sgld_steps):
        print(f"{step+1} of {sgld_steps} steps")
        x_k.requires_grad = True
        d_model_dx = torch.autograd.grad(model(x_k).sum(), x_k, retain_graph=True)[0] # TODO: remove retain graph=TRUE
        x_k = x_k.detach()
        x_k += sgld_step_size * d_model_dx + sgld_noise * torch.randn_like(x_k)

    sgld_samples = x_k.detach()

    return sgld_samples


def create_random_buffer(size, n_channels, im_size):
    return torch.FloatTensor(size, n_channels, im_size, im_size).uniform_(-1, 1)


def load_model_and_buffer(load_dir, device):
    print(f"loading model and buffer from {load_dir} ...")
    model = wide_resnet_energy_output.WRN_Energy(params["depth"], params["widen_factor"], 0.0, 10)
    checkpoint_dict = torch.load(load_dir)
    model.load_state_dict(checkpoint_dict["model"])
    model = model.to(device)
    buffer = checkpoint_dict["buffer"]

    return model, buffer


def main(save_dir, model_load_dir, sgld_step_size, sgld_noise, sgld_steps=10):

    sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
    plot = lambda path, x: torchvision.utils.save_image(torch.clamp(x, -1, 1), path, normalize=True, nrow=sqrt(x.size(0)))

    # create the save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    x_k = create_random_buffer(100, 3, 32)
    plot(f"{save_dir}initial_start.png", x_k)
    x_k = x_k.to(device)

    model, _ = load_model_and_buffer(model_load_dir, device)
    model = model.to(device)

    sgld_samples = run_fresh_sgld(model, x_k, sgld_steps, sgld_step_size, sgld_noise)

    plot(f"{save_dir}fresh_sgld_{sgld_steps}.png", sgld_samples)

    sgld_cpu_samples = {"buffer": sgld_samples.cpu()}
    torch.save(sgld_cpu_samples, f"{save_dir}generated_samples_{sgld_steps}.pt")


if __name__ == "__main__":

    # specify generated sample save path and model load path
    save_path = params["save_path"]
    save_dir = f"{save_path}fresh_sgld_samples/"
    model_load_path = f"{save_path}ckpt_145_epochs.pt"

    main(save_dir=save_dir,
         model_load_dir=model_load_path,
         sgld_step_size=params["sgld_step_size"],
         sgld_noise=params["sgld_noise"])




