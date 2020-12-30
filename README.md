# ML Reproducibility Challenge 2020: Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One

This repo contains a re-implemimentation of the 2020 ICLR paper [Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One](https://arxiv.org/abs/1912.03263). A reproducibility report, submitted to the 2020 ML reproducibility challenge, is available [here](https://openreview.net/forum?id=ShrPBsjByVa&referrer=%5BML%20Reproducibility%20Challenge%202020%5D(%2Fgroup%3Fid%3DML_Reproducibility_Challenge%2F2020)).
Local code was run on Mac OS X Mojave, version 10.14.6

## Installation

1. Install [conda](https://docs.anaconda.com/anaconda/install/)
1. Create conda environment:
    `conda env create -f environment.yml`
1. Activate environment
    `conda activate ml_reprod_hybrid_energy_models`
1. Confirm you can run the training scripts
    `python train_supervised.py` and `python train_JEM_algorithm.py`

## Training

* To train a model with the supervised training method run `python train_supervised.py`. Model artefacts (checkpoints) will be stored in `./artefacts_supervised`.
* To train a model with the joint energy-based model (JEM) training method, run `python train_JEM_algorithm.py`. Model artefacts (checkpoints, images) will be stored in `./artefacts`. _Note_: The JEM training technique is unstable, and the training run will crash. This is discussed in Appendix H.3 of the [paper](https://arxiv.org/abs/1912.03263). If and when it does crash, do the following: In `params.json`, change `params["load_from_checkpoint"]` to `True`, and change `params["start_epoch"]` to the epoch where it crashed. Try loading from earlier checkpoints if the most recent one crashes. Re-run, and repeat.

## Analysis

* Once the model is trained, run `python calibration.py` to generate the calibration plots for both the supervised as well as JEM training methods. An example calibration plot is shown in `artefacts`.
* To generate fresh SGLD samples (from a randomly initialized buffer), run `python generate_samples.py`. By default, it will run for 20 SGLD steps but this can be changed. Example SGLD evolutions for 1, 5, 10, 20 and 50 steps are given in `artefacts/fresh_sgld_samples`.
