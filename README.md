# Palette View Synthesis: Novel View Synthesis using Diffusion Probabilistic Modelling

## Summary

Novel view synthesis is a class of computer vision problems, in which one or multiple views of a scene or an object are provided. The goal is then to produce novel, previously unseen views of the given scene or object. Recently, the endeavors to solve such problems have gained significant traction in the generative deep learning domain. From Neural Radiance Field (NeRF) based approaches to encoder-decoder style architectures, various ways of performing novel view synthesis have been previously introduced. <br> <br>
This work introduces Palette View Synthesis, an end-to-end diffusion probabilistic generative modelling approach for performing novel view synthesis which aims to resolve the drawbacks of previous approaches by extending the model's abilities to generalize across multiple classes, given only a single view and a target angle of the object as inputs, while simultaneously maintaining the quality of the generated samples. It shows that by employing a diffusion-based model, with a simple U-Net backbone that parameterizes the denoising function, and conditioning via concatenation along the input channel dimension, it is possible to produce high quality, believable novel views while simultaneously generalizing across multiple different classes.

## Model 

## Dataset

## Results

## Usage


## Citation


