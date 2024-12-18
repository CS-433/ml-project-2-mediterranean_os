import os
import random

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from src.dataset import MyDataset
from src.metrics import (Bce_plus_DiceLoss, DiceLoss, calculate_metrics,
                         plot_images)
from src.models.model_unet3p import unet3p
from src.models.model_unet3p_Resnet import unet3p_back
from src.models.UNet import UNet
from src.models.UNet_Resnet import UnetResnet34

LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 6
CLIP_VALUE = 5.0
NUM_EPOCHS = 20
NUM_WORKERS = 2
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
PIN_MEMORY = True
RESUME_TRAINING = False
MASSACHUSSETS = False
TRAIN_IMG_DIR = "./training" 
VAL_IMG_DIR = "./test_set_images"


def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def debug_check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN detected in gradients of {name}")

best_dice = 0
best_threshold = 0.5
def train(train_loader, test_loader, model, optimizer, scheduler, criterion, epoch):
    
    model.train(mode = True)
    
    train_loss_history = []
    val_loss_history = []
    lr_history = []
    
    total_iou_train = 0.0
    total_pixel_accuracy_train = 0.0
    total_dice_coefficient_train = 0.0
    train_loss = 0.0
    global best_threshold

    train_loader = tqdm(train_loader, desc=f"Epch {epoch +1}/{NUM_EPOCHS}",unit="batch")

    for batch_idx, (images, masks) in enumerate(train_loader):

        images = images.to(device = DEVICE)
        masks = masks.float().to(device = DEVICE)        

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss_history.append(loss.item())
        train_loss = loss.item()
        lr_history.append(scheduler.get_last_lr()[0])

        with torch.no_grad():
            outputs_prob = torch.sigmoid(outputs)
            best_dice_threshold = 0
            best_iou_train = 0 
            best_dice_coefficient_train = 0
            best_pixel_accuracy_train = 0
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                pred_masks = outputs_prob > threshold
                iou_train, dice_coefficient_train, pixel_accuracy_train = calculate_metrics(
                            pred_masks, masks)
                if (dice_coefficient_train > best_dice_threshold):
                    best_dice_threshold = dice_coefficient_train
                    best_threshold = threshold
                    best_iou_train = iou_train
                    best_dice_coefficient_train = dice_coefficient_train
                    best_pixel_accuracy_train = pixel_accuracy_train
            
            current_lr = optimizer.param_groups[0]["lr"]

            total_iou_train += best_iou_train
            total_dice_coefficient_train+= best_dice_coefficient_train
            total_pixel_accuracy_train+= best_pixel_accuracy_train
        train_loader.set_postfix(
                    train_iou=best_iou_train,
                    train_pix_acc=best_pixel_accuracy_train,
                    train_dice_coef=best_dice_coefficient_train,
                    lr = current_lr
                )
        
    #iou(intersection over union) evaluate the performance of object detection
    avg_iou_train = total_iou_train / len(train_loader)
    #gives the proportion of correctly labeled pixels out
    avg_pixel_accuracy_train = total_pixel_accuracy_train / len(train_loader)
    #dice coefficient is a measure of similarity bewtween two sets
    avg_dice_coefficient_train = total_dice_coefficient_train / len(train_loader)

    model.eval()
    val_loss = 0.0
    total_iou_val = 0.0
    total_pixel_accuracy_val = 0.0
    total_dice_coefficient_val = 0.0

    test_loader = tqdm(test_loader, desc=f"Validation", unit="batch")

    #debug_check_gradients(model)

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)

            v_loss = criterion(outputs, masks)
            val_loss += v_loss.item()
            pred_masks = outputs > best_threshold
            iou_val, dice_coefficient_val, pixel_accuracy_val = calculate_metrics(
                pred_masks, masks
            )

            total_iou_val += iou_val
            total_pixel_accuracy_val += pixel_accuracy_val
            total_dice_coefficient_val += dice_coefficient_val

            test_loader.set_postfix(
                        val_loss=v_loss.item(),
                        val_iou=iou_val,
                        val_pix_acc=pixel_accuracy_val,
                        val_dice_coef=dice_coefficient_val,
                        lr=current_lr,
                    )

    val_loss /= len(test_loader)
    avg_iou_val = total_iou_val / len(test_loader)
    avg_pixel_accuracy_val = total_pixel_accuracy_val / len(test_loader)
    avg_dice_coefficient_val = total_dice_coefficient_val / len(test_loader)

    val_loss_history.append(val_loss)

    global best_dice

    if avg_dice_coefficient_val > best_dice:
        best_epoch = epoch
        best_dice = avg_dice_coefficient_val
        checkpoint(model, 'model_state_dict.pt')

    print(  
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS}\n"
        f"Avg Validation Loss: {val_loss:.4f}\n"
        f"Avg Train Loss: {train_loss:.4f}\n"
        f"Avg IoU Train: {avg_iou_train:.4f}\n"
        f"Avg IoU Val: {avg_iou_val:.4f}\n"
        f"Avg Pix Acc Train: {avg_pixel_accuracy_train:.4f}\n"
        f"Avg Pix Acc Val: {avg_pixel_accuracy_val:.4f}\n"
        f"Avg Dice Coeff Train: {avg_dice_coefficient_train:.4f}\n"
        f"Avg Dice Coeff Val: {avg_dice_coefficient_val:.4f}\n"
        f"Best Threshold: {best_threshold}\n"
        f"Current LR: {current_lr}\n"
        f"{'-'*50}"
    )
    return train_loss_history, val_loss_history, lr_history

def main():
    set_seed(seed = 42) 
    model = UnetResnet34(num_classes = 1, input_size = IMAGE_WIDTH).to(device = DEVICE)
    if (RESUME_TRAINING):
        resume(model, 'UnetResnet.pt')
        
    space_transforms = A.Compose(
        [
            A.Rotate(limit=(-90, 90), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CoarseDropout(num_holes_range = (3,6), hoxle_height_range = (8,15), hole_width_range = (8,15), p = 1.0),
            A.ElasticTransform(p =1.0),
        ]
    )
    colour_transforms = A.Compose(
        [
            A.ColorJitter(brightness=0.4, contrast = 0.4, hue = 0.2, p = 0.5, ),
            A.ColorJitter(saturation=0.4, p = 0.5),
        ]   
    )

    print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Should print your GPU name

    pct = 0.2
    if (MASSACHUSSETS):
        train_dataset = MyDataset("./training_extra", space_transforms, colour_transforms, 'train', pct)
        test_dataset = MyDataset("./training_extra", None, None, 'test', pct)
    else:
        train_dataset = MyDataset(TRAIN_IMG_DIR, space_transforms, colour_transforms, 'train', pct)
        test_dataset = MyDataset(TRAIN_IMG_DIR, None, None, 'test', pct)

    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle = True)
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle = True)

    optimizer = torch.optim.AdamW(model.parameters(), amsgrad = True, lr = LEARNING_RATE, weight_decay=1e-5)   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max= 2000 )
    criterion = Bce_plus_DiceLoss()
    
    lr_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(NUM_EPOCHS):
        train_loss, val_loss, lrs = train(train_dataloader, test_dataloader, model, optimizer, scheduler, criterion, epoch)
        train_loss_history.extend(train_loss)
        val_loss_history.extend(val_loss)
        lr_history.extend(lrs)
        
    n_train = len(train_loss_history)
    n_val = len(val_loss_history)
    t_train = NUM_EPOCHS * np.arange(n_train) / n_train
    t_val = NUM_EPOCHS * np.arange(n_val) / n_val

    plt.figure(figsize=(6.4 * 3, 4.8))
    plt.subplot(1, 3, 1)
    plt.plot(t_train, train_loss_history)
    plt.xlabel("Test loss")

    plt.subplot(1,3,2)
    plt.plot(t_val, val_loss_history, color = 'r')
    plt.xlabel("Validation loss")

    plt.subplot(1, 3, 3)
    plt.plot(t_train, lr_history)
    plt.xlabel("Epoch")

    plt.show()


if __name__ == "__main__":
    main()
    