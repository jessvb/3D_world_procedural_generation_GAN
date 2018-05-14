# Procedural Generation of 3D Worlds
This repository contains GAN and style transfer code to generate 3D worlds.

## GAN versions
Each version of the GAN has incremental updates that change the way the GAN works. They each create quite varied landscapes; thus, each version is kept separately in the repo so that anyone can use any of the GAN networks at any time. The following list describes the updates at each iteration (in lieu of a git log ;) ):
* v0: a basic GAN based on the one found here: https://github.com/uclaacmai/Generative-Adversarial-Network-Tutorial
* v2.1: uses leaky ReLU activations in discriminator network instead of regular ReLU and gives a learning rate seed to the adam optimizer
* v3: uses resize upscaling to remove checkerboarding artefacts (as outlined here: https://distill.pub/2016/deconv-checkerboard/)
* v4: uses two different learning rates for the discriminator and generator networks
* v4.1: makes the GAN train for an infinite number of iterations
* v4.2: uses input images of 256x256px
* v4.3: changes all tf.nn.relu activations in the generator to tf.nn.leaky_relu and scales the input image values to between 1 and 2 (instead of -1 and 1) to prevent values from getting stuck at zero
* v5: returns to transpose upscaling with a 5x5px filter size (to see if the checkerboard will disappear in this case too)
* v6: updates filter size to 4x4 in addition to different discriminator and generator learning rates
* v6.1: allows for variable filter size
* v7.0: returns to image resize upscaling, and includes the variable filter size
* v8.0: decreases the number of convolutional layers to reduce the chance of overfitting
* v9.0: changes the prediction probabilities slightly
* v10.0: fixes a discriminator loss bug by resetting the input images to begin from the start once they have all been fed into the GAN (and reverts the prediction probabilities to the previous version before 9.0)

## Style Transfer Network Files

TODO: insert description of how the style transfer network files all work together

## Useful Scripts
This folder contains scripts to help create a clean dataset.
delDarkUnvariedIdentical.py: deletes images that are below a certain lightness threshold, have low variation (e.g., all white or all black), or are identical to other images placed in a folder called "duplicates"
* prepDataset.py: pickles images, preparing them to be input images for the GAN
* scaleToWidthxWidth: scales images to be a certain size
