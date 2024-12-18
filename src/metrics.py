import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Any

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOOTH = 1e-5

class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()

    def forward(self, pred_mask: Any, true_mask: Any) -> torch.Tensor:
        intersection = torch.sum(pred_mask * true_mask)
        union = torch.sum(pred_mask) + torch.sum(true_mask)

        # Add a small epsilon to the denominator to avoid division by zero
        dice_loss = 1.0 - (2.0 * intersection + SMOOTH) / (union + SMOOTH)
        return dice_loss
    
class Bce_plus_DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(Bce_plus_DiceLoss, self).__init__()

    def forward(self, output: Any, true_mask: Any) -> torch.Tensor:
        bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=(torch.FloatTensor ([0.225])).to(device=DEVICE))(output, true_mask)
        
        pred_mask = torch.sigmoid(output)
        intersection = torch.sum(pred_mask * true_mask)
        union = torch.sum(pred_mask) + torch.sum(true_mask)
        dice_loss = 1.0 - (2.0 * intersection + SMOOTH) / (union + SMOOTH)

        return bce_loss + dice_loss    
    
def calculate_metrics(pred_mask: Any, true_mask: Any) -> torch.Tensor:
    pred_mask = pred_mask.float()
    true_mask = true_mask.float()

    intersection = torch.sum(pred_mask * true_mask)
    union = torch.sum((pred_mask + true_mask) > 0.5)

    # Add a small epsilon to the denominator to avoid division by zero
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    dice_coefficient = (2 * intersection + SMOOTH) / (
        torch.sum(pred_mask) + torch.sum(true_mask) + SMOOTH
    )
    pixel_accuracy = torch.sum(pred_mask == true_mask) / true_mask.numel()

    return iou.item(), dice_coefficient.item(), pixel_accuracy.item()

def plot_images(images, masks, outputs, n_image_rows):
    plt.figure(figsize=(6.4 * 3, 4.8 * n_image_rows))
    for i in range(n_image_rows):
        plt.subplot(n_image_rows, 3, 3 * i + 1)
        plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(n_image_rows, 3, 3 * i + 2)
        plt.imshow(masks[i].permute(1, 2, 0).cpu().numpy() ,cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(n_image_rows, 3, 3 * i + 3)
        plt.imshow(outputs[i].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
        plt.title("Output")
        plt.axis("off")

    plt.show()


    plt.show()
