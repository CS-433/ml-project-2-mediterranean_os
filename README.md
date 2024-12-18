# ML_project_EPFL
> - Semantic road segmentation project for CS-433 Machine Learning at EPFL
> - Prediction model for semantic road segmentation of satellite images created by Team "mediterranean_OS"

## Table of Contents
1. [Introduction](#introduction)
2. [Datasets](#datasets)
3. [Setup](#setup)
4. [Project Structure](#project-structure)
5. [Usage](#usage)
6. [Authors](#authors)

## Introduction
This project implements a UNet-based deep learning model to address the problem of semantic road segmentation. Semantic segmentation is critical for applications such as autonomous driving and urban planning, enabling precise identification of road regions in images. Apart from the basic UNet architecture, UNet3+ and a RESNet34 backbone have been added to compare the performance of predictions against the base implementation.
<br />
For this project, we used an already implemented UNet3+: [Link to original repository](https://github.com/nikhilroxtomar/UNET-3-plus-Implementation-in-TensorFlow-and-PyTorch)

## Datasets
The dataset used for training is made up of 100 different satellite images (400x400px) of american style suburbs, along with their respective ground truth images highlighting the roads that are used during the supervised learning. For testing, the dataset is comprised by 50 satellite images (608x608px) of similar american style suburbs. 
<br /><br />
![Satellite image](https://github.com/CS-433/ml-project-2-mediterranean_os/blob/main/resources/satImage_001.png)
![Ground truth image](https://github.com/CS-433/ml-project-2-mediterranean_os/blob/main/resources/satImage_001_ground.png)

## Setup
- Clone the repository with `git clone https://github.com/CS-433/ml-project-2-mediterranean_os.git`
- This project uses Pytorch with CUDA, in case you installed Pytorch without CUDA enabled, you will need to reinstall Pytorch following [this short guide](https://pytorch.org/get-started/locally/)

## Project Structure
Generate tree

## Usage
**Hyperparameter tuning:**
Multiple parameters in `train.py` can be adjusted to fine tune the performance of the model. These parameters are:
- `BATCH_SIZE`: the size of the mini-batches used by the model during training, smaller mini-batches reduce computational cost but can impact performance.
- `NUM_EPOCHS`: number of training cycles the model will go through.
- `LEARNING_RATE`: controls the step size at which the optimization algorithm updates model weights during training.

**Choosing different models:**
Different models can be trained and later used to make predictions, the available models can be found in `/models`. Changing the model can be done by importing the model in `train.py` and using it at the top of the `main()` function in `train.py`.:
- **[UNet:](https://arxiv.org/abs/1505.04597)** UNet description 
- **[UNet3+:](https://arxiv.org/abs/2004.08790)** description
- **[RESNet34 backbone:](https://arxiv.org/abs/1512.03385)** both models can be run with a RESNet34 backbone to improve performance at the cost of more computation required. 

**Running the program:**
Run the code in `train.py` and wait for the results and the visualizations.

## Authors
| Student's name | SCIPER |
| -------------- | ------ |
| Leonardo Denovellis | 386074 |
| Luis Bustamante | 366705 |
| Antonio Silvestri | 376875 |
