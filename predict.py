import os
from glob import glob

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

import src.mask_to_submission as mts
from src.dataset import SubmissionDataset
from src.metrics import DiceLoss, calculate_metrics, plot_images
from src.models.UNet3p_Resnet import UNet3p_Resnet
from src.models.UNet import UNet
from src.models.UNet_Resnet import Unet_Resnet
from src.models.UNet3p import UNet3p

IMAGE_HEIGHT = 608
IMAGE_WIDTH = 608
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
TRAIN_IMG_DIR = "./training" 
VAL_IMG_DIR = "./test_set_images"

if __name__ == "__main__":
    model = UnetResnet34( input_size = IMAGE_WIDTH, num_classes = 1).to(device = DEVICE)
    
    model.load_state_dict(torch.load('model_state_dict.pt'))

    model.eval()  
    eval_transform = A.Compose(
        [
            ToTensorV2(),
        ]
    )

    evalDataset = SubmissionDataset(VAL_IMG_DIR, transform = eval_transform)
    evalDataloader = DataLoader(evalDataset, batch_size = 1, shuffle = False)

    OVERWRITE_FOLDER = True
    if OVERWRITE_FOLDER:
        os.makedirs('submissions', exist_ok=True)
        for f in glob('./submissions/*'):
            os.remove(f)
    
    image_filenames = []
    with torch.no_grad():
        for i, (image, index) in enumerate(evalDataloader):
            image = image.to(DEVICE)
            output = model.forward(image)               
            prev_mask = (output > 0.1).float()

            image_filename = 'submissions/test_' + '%.3d' % (index+1) + '.png'
            save_image(prev_mask[0], image_filename)
            image_filenames.append(image_filename)
    
    submission_filename = "test.csv"
    mts.masks_to_submission(submission_filename, *image_filenames)


