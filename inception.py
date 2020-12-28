import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from tqdm import tqdm

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

# Inception score code adapted from https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py


def obtain_top_k(preds, k):
    """
    :param preds: (arr) array of predictions (shape (N_imgs, N_classes))
    :param k: (int) top k most confident predictions
    :return: (arr) top_k_predictions (shape (k, N_classes)), top_k_inds (shape (k,))
    """
    max_preds = np.amax(preds, axis=1)
    top_k_inds = np.argpartition(max_preds, -k)[-k:]
    top_k_preds = preds[top_k_inds]

    return top_k_preds, top_k_inds


def ensemble_buffer(buffers):
    """
    :param buffers: (list[arr]) list of buffers
    :return: (arr) ensembled buffer
    """
    stacked_buffers = torch.stack(buffers)

    return torch.mean(stacked_buffers, dim=0)


def obtain_inception_predictions(imgs, cuda=True, batch_size=32, resize=False, save_preds=False):
    """Computes the inception model predictions of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    total = len(imgs) // batch_size if len(imgs) % batch_size == 0 else len(imgs) // batch_size + 1

    for i, batch in tqdm(enumerate(dataloader), total=total):

        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    if save_preds:
        with open("inception_preds_ensembled.npy", "wb") as f:
            np.save(f, preds)

    return preds


def inception_score(preds, splits=1):
    """
    :param preds: (arr) np array of network predictions (shape: (N_imgs, N_classes)
    :param splits: (int) number of times to split predictions list
    :return: (float) score
    """

    # Now compute the mean kl-div
    N = len(preds)
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
