# EMLO Session 11

The goal of the assignment is to detect data drift on a trained model.

## Training Log - Tensorboard
The logs can be viewed [here](https://tensorboard.dev/experiment/8hpOBqj3RVmkxDr1u6RS6A/)

25 images have been logged on each epoch. Unfortunately, these figures do not appear on the tensorboard-dev website. They can be viewed locally using `tensorboard --logdir ./ --bind_all`.

Note that for logging purpose, the normalization transform has not been performed since it distorts the colors of the image; it has however been used for training and testing the model.


## Data Drift

For drift, we checked the Maximum Mean Discrepancy using the `alibi-detect` library. For a randomly chose image, it didn't detect data drift on the perturbed image (probably because the model was trained using augmentations).

On a set of 100 images, it did detect data drift.

Please see the [notebook](./code/assign.ipynb)