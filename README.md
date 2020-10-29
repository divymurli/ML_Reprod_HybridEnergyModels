# ML Reproducibility Challenge 2020: Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One

This repo contains a re-implemimentation of the 2020 ICLR paper [Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One](https://arxiv.org/abs/1912.03263).
Local code was run on Mac OS X Mojave, version 10.14.6

## Installation

1. Install [conda](https://docs.anaconda.com/anaconda/install/)
1. Create conda environment:
    `conda env create -f environment.yml`
1. Activate environment
    `conda activate ml_reprod_hybrid_energy_models`
1. Confirm you can run the training script
    `python train_supervised.py`