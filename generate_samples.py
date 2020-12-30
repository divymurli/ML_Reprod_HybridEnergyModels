import json
import os
import torch

import torchvision
from models import wide_resnet_energy_output
from utils import create_random_buffer, run_sgld, load_model_and_buffer

dir_path = os.path.dirname(os.path.realpath(__file__))
p = os.path.join(dir_path, 'params.json')
with open(p, 'r') as f:
    params = json.load(f)


def main(save_dir, model_load_dir, sgld_step_size, sgld_noise, sgld_steps=50):

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

    architecture = wide_resnet_energy_output.WRN_Energy(params["depth"], params["widen_factor"], 0.0, 10)

    model, _ = load_model_and_buffer(model_load_dir, architecture, device)
    model = model.to(device)
    model.eval()
    sgld_samples = run_sgld(model, x_k, sgld_steps, sgld_step_size, sgld_noise, print_step=True)
    sgld_samples = sgld_samples.detach()
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
         sgld_noise=params["sgld_noise"],
         sgld_steps=20)




