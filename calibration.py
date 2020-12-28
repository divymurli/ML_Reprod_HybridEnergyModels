import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from models import wide_resnet, wide_resnet_energy_output
from utils import load_model_and_buffer

plt.rcParams.update({'font.size': 25})

dir_path = os.path.dirname(os.path.realpath(__file__))
p = os.path.join(dir_path, 'params.json')
with open(p, 'r') as f:
    params = json.load(f)


# IMAGE CHARACTERISTICS #
# define image parameters
n_channels = 3
im_size = 32


# DATA LOADING AND AUGMENTATION #
# normalize all pixel values to be in [-1, 1] and add Gaussian noise with mean zero, variance gaussian_noise_var
# using the same train/test data augmentation as in the paper's code

transform_test = transforms.Compose(
            [transforms.ToTensor(),
             # normalize by the mean, stdev of the training set
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2488, 0.2453, 0.2633)),
             lambda x: x + params["gaussian_noise_var"] * torch.randn_like(x)]
)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=params["test_batch_size"],
                                         shuffle=False, num_workers=2)


def correct_and_confidence(model, loader, device, with_energy=True):

    """
    return the correctness and confidences of predictions
    :param model: (obj) model
    :param loader: (iter) train or test loader
    :param with_energy: (bool) use energy or not
    :return: (arr) array of shape (examples, 2) of correct outputs and confidences
    """

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
            confidence = confidence.float().cpu().numpy()
            confidences.extend(confidence)
            correct = (predicted == labels).float().cpu().numpy()
            corrects.extend(correct)

    return np.array(sorted(list(zip(corrects, confidences)), key=lambda x: x[1]))


def calibration_buckets(zipped_corr_conf):

    """
    return calibration buckets
    :param zipped_corr_conf: (arr)
    :return:
            (list) bucket boundaries
            (list) averaged accuracy of examples within each bucket
            (list) averaged confidence of examples within each bucket
            (list) number of examples in each bucket
    """

    thresholds = np.linspace(0, 1, 21)
    corrects = zipped_corr_conf[:, 0]
    confidences = zipped_corr_conf[:, 1]

    buckets = [(thresholds[i], thresholds[i+1]) for i in range(len(thresholds) - 1)]
    bucket_accs = []
    bucket_confs = []
    bucket_totals = []

    for bucket in buckets:
        total = 0
        correct = 0
        conf = 0
        for i in range(len(confidences)):
            if confidences[i] > bucket[0] and confidences[i] < bucket[1]:
                total += 1
                correct += corrects[i]
                conf += confidences[i]
        if total != 0:
            bucket_acc = correct / total
            bucket_conf = conf / total
        else:
            bucket_acc = 0.
            bucket_conf = 0.
        bucket_accs.append(bucket_acc)
        bucket_confs.append(bucket_conf)
        bucket_totals.append(total)

    return buckets, bucket_accs, bucket_confs, bucket_totals


def expected_calibration_error(data_length, bucket_accs, bucket_confs, bucket_totals):

    """
    compute expected calibration error (ECE)
    :param data_length: (int) number of examples
    :param bucket_accs: (list) averaged accuracy in each bucket
    :param bucket_confs: (list) averaged confidence in each bucket
    :param bucket_totals: (list) number of examples in each bucket
    :return: (float) ECE
    """

    normalization = (1/data_length)*np.array(bucket_totals)
    ece = np.abs(np.array(bucket_accs) - np.array(bucket_confs))

    ece = np.dot(normalization, ece)

    return ece


def main(load_dir_JEM, load_dir_sup):
    # Analysis

    # define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # define model
    JEM_architecture = wide_resnet_energy_output.WRN_Energy(params["depth"], params["widen_factor"], 0.0, 10)
    supervised_architecture = wide_resnet.WideResNet(params["depth"], params["widen_factor"], 0.0, 10)

    print("loading JEM and supervised models ...")
    model_JEM, buffer = load_model_and_buffer(load_dir_JEM, JEM_architecture, device)
    model_sup = load_model_and_buffer(load_dir_sup, supervised_architecture, device, with_energy=False)

    print("computing calibrations ...")
    zipped_corr_conf_JEM = correct_and_confidence(model_JEM, testloader, device)
    zipped_corr_conf_sup = correct_and_confidence(model_sup, testloader, device, with_energy=False)

    print("saving intermediate calibrations ...")
    with open("zipped_corr_conf.npy", "wb") as f:
        np.save(f, zipped_corr_conf_JEM)

    with open("zipped_corr_conf_supervised.npy", "wb") as f:
        np.save(f, zipped_corr_conf_sup)

    zipped_corr_conf_sup = np.load("zipped_corr_conf_supervised.npy")
    zipped_corr_conf = np.load("zipped_corr_conf.npy")
    buckets, bucket_accs, bucket_confs, bucket_totals = calibration_buckets(zipped_corr_conf_sup)
    buckets, bucket_accs_JEM, bucket_confs_JEM, bucket_totals_JEM = calibration_buckets(zipped_corr_conf)
    ticklabels = [round(i, 1) for i in np.linspace(0, 1, 6)][:-1]

    fig = plt.figure(figsize=(20, 10), facecolor="white")

    ax = fig.add_subplot(121)
    ax.bar(np.arange(20), height=bucket_accs)
    ax.set_xticks(np.arange(0, 20, 4))
    ax.set_xticklabels(ticklabels)
    ax.set_ylim(0, 1)
    x = np.linspace(*ax.get_xlim())
    y = np.linspace(*ax.get_ylim())
    ax.plot(x, y, linestyle='dashed', color='red')
    ax.set_xlabel("bucket")
    ax.set_ylabel("bucket accuracy")
    ax.set_title("Ordinary Supervised", pad=20)

    ax2 = fig.add_subplot(122)
    ax2.bar(np.arange(20), height=bucket_accs_JEM)
    ax2.set_xticks(np.arange(0, 20, 4))
    ax2.set_xticklabels(ticklabels)
    ax2.set_ylim(0, 1)
    x = np.linspace(*ax2.get_xlim())
    y = np.linspace(*ax2.get_ylim())
    ax2.plot(x, y, linestyle='dashed', color='red')
    ax2.set_xlabel("bucket", )
    ax2.set_ylabel("bucket accuracy")
    ax2.set_title("JEM", pad=20)

    fig.savefig("./artefacts/calibration_plots.png")

    ece_sup = expected_calibration_error(10000, bucket_accs, bucket_confs, bucket_totals)
    ece_JEM = expected_calibration_error(10000, bucket_accs_JEM, bucket_confs_JEM, bucket_totals_JEM)
    print(f"ece_sup: {ece_sup}")
    print(f"ece_JEM: {ece_JEM}")


if __name__ == "__main__":

    # specify directory to load models
    # TODO: put these inside params.json
    load_dir_JEM = "./artefacts/ckpt_145_epochs.pt"
    load_dir_sup = "./artefacts/model_145_epochs.pt"

    main(load_dir_JEM=load_dir_JEM, load_dir_sup=load_dir_sup)





