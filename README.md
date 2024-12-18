# ML_project_EPFL
> - Semantic road segmentation project for CS-433 Machine Learning at EPFL
> - Prediction model for semantic road segmentation of satellite images created by Team "mediterranean_OS"

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Authors](#authors)

## Introduction
This project implements a UNet-based deep learning model to address the problem of semantic road segmentation. Semantic segmentation is critical for applications such as autonomous driving and urban planning, enabling precise identification of road regions in images. Apart from the basic UNet architecture, UNet3+ and a RESNet34 backbone have been added to compare the performance of predictions against the base implementation. <br />
The dataset used for training is made up of 100 different satellite images (400x400px) of american style suburbs, along with their respective ground truth images highlighting the roads that are used during the supervised learning. For testing, the dataset is comprised by 50 satellite images (608x608px) of similar american style suburbs.


## Setup
- Clone the repository with `git clone https://github.com/CS-433/ml-project-2-mediterranean_os.git`
- Something else

## Project Structure
Generate tree

## Usage
**Hyperparameter tuning:**
Multiple parameters in `train.py` can be adjusted to fine tune the performance of the model. These parameters are:
- `batch_size`: the size of the mini-batches used ...
- others

**Choosing different models:**
Different models can be trained and later used to make predictions, the available models can be found in `/models` and are:
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
