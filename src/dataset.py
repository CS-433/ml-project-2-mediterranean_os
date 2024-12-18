import os
import random
from glob import glob

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class MyDataset(Dataset):
    def __init__(self, data_dir, space_transforms = None, colour_transforms = None, data_type = 'train', pct = .9 ):
       
        self.data_dir = data_dir
        self.space_transforms = space_transforms
        self.colour_transforms = colour_transforms

        self.data_type = data_type
        self.pct = pct

        self.images_dir = os.path.join(data_dir, "images")
        self.groundtruth_dir = os.path.join(data_dir, "groundtruth")
        self.image_filenames = sorted(glob(os.path.join(self.images_dir, "*.png")))
        self.mask_filenames = sorted(glob(os.path.join(self.groundtruth_dir, "*.png")))
        num_samples = len(self.image_filenames)
        indices = list(range(num_samples))
        random.shuffle(indices)  # Shuffle the indices randomly
        num_test_samples = int(num_samples * self.pct)

        if self.data_type == "train":
                if num_test_samples == 0:
                    self.indices = indices
                else:
                    self.indices = indices[:-num_test_samples]
        elif self.data_type == "test":
                if pct == 0:
                    self.indices = []
                else:
                    self.indices = indices[-num_test_samples:]

    def __len__(self):  
        return len(self.indices)
    
    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        mask_name = self.mask_filenames[index]

        image = np.array(Image.open(img_name).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(mask_name).convert("L"), dtype=np.float32)   # L = grey scale

        image = image / 255.0 
        mask = mask / 255.0

        if self.space_transforms is not None:
            augmentations = self.space_transforms(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]   

        if self.colour_transforms is not None:
            augmentations = self.colour_transforms(image = image)
            image = augmentations["image"]

        totensor = A.Compose(
            [
                ToTensorV2()
            ]
        )(image = image, mask = mask)
        image = totensor["image"]
        mask = totensor["mask"]

        mask = np.expand_dims(mask, axis=0)

        return image, mask
 
class MassachussetsDataset(Dataset):
    def __init__(self, data_dir, space_transforms = None, colour_transforms = None, 
                 data_type = 'train', pct = .9, max_size = 50 ):
       
        self.data_dir = data_dir
        self.space_transforms = space_transforms
        self.colour_transforms = colour_transforms

        self.data_type = data_type
        self.pct = pct

        self.images_dir = os.path.join(data_dir, "train")
        self.groundtruth_dir = os.path.join(data_dir, "train_labels")
        self.image_filenames = sorted(glob(os.path.join(self.images_dir, "*.tiff")))[:max_size]
        self.mask_filenames = sorted(glob(os.path.join(self.groundtruth_dir, "*.tiff")))[:max_size]

        num_samples = len(self.image_filenames)
        indices = list(range(num_samples))
        num_test_samples = int(num_samples * self.pct)

        if self.data_type == "train":
                if num_test_samples == 0:
                    self.indices = indices
                else:
                    self.indices = indices[:-num_test_samples]
        elif self.data_type == "test":
                if pct == 0:
                    self.indices = []
                else:
                    self.indices = indices[-num_test_samples:]

    def __len__(self):  
        return len(self.indices)
    
    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        mask_name = self.mask_filenames[index]

        image = np.array(Image.open(img_name).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(mask_name).convert("L"), dtype=np.float32)   # L = grey scale

        image = image / 255.0 
        mask = mask / 255.0

        if self.space_transforms is not None:
            augmentations = self.space_transforms(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]   

        if self.colour_transforms is not None:
            augmentations = self.colour_transforms(image = image)
            image = augmentations["image"]

        
        totensor = A.Compose(
            [
                A.Crop(x_min = 0, x_max = 400, y_min = 0, y_max = 400, p = 1),
                ToTensorV2()
            ]
        )(image = image, mask = mask)
        image = totensor["image"]
        mask = totensor["mask"]


        mask = np.expand_dims(mask, axis=0)

        return image, mask    
    
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        # Handle independent dataset lengths with modulo indexing
        return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        # Choose length based on the dataset with the maximum number of samples
        return max(len(d) for d in self.datasets)

def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    numeric_part = ''.join(filter(str.isdigit, filename))
    return int(numeric_part)

class SubmissionDataset(Dataset):
    def __init__(self, data_dir, transform = None):
       
        self.data_dir = data_dir
        self.transform = transform

        self.images_dir = os.path.join(data_dir)
        self.image_filenames = sorted(glob(os.path.join(data_dir, "*.png")), key=get_key)

        self.num_samples = len(self.image_filenames)
        self.indices = list(range(self.num_samples))

    def __len__(self):  
        return len(self.indices)
    
    def __getitem__(self, index):
        img_name = self.image_filenames[index]

        image = np.array(Image.open(img_name).convert("RGB"), dtype=np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        image = image / 255.0 

        return image, index

def visualize(image, mask):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(image)
    ax.set_title('Image')
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(mask)
    ax.set_title('Label')

    plt.show()

    
if __name__ == "__main__":
    data_dir = "./training"
    input_size = (400,400)

    transform = A.Compose(
        [
            A.Rotate(limit=(-10, 10), p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomCrop(height=input_size[0], width=input_size[0], p=0.7),
            ToTensorV2(),
        ]
    )

    train_dataset = MyDataset(data_dir, transform, 'train', pct = 0.75)
    train_dataloader = DataLoader(train_dataset, batch_size = 3, shuffle = False)
    test_dataset = MyDataset(data_dir, transform, 'test', pct = 0.75)
    test_dataloader = DataLoader(test_dataset, batch_size = 3, shuffle = False)

    for batch_id, (images, masks) in enumerate(test_dataloader):
        if batch_id == 0:
            for i in range(images.shape[0]):
                view_img = images[i].swapaxes(0,1)
                view_img = view_img.swapaxes(1,2)
                view_mask = masks[i].swapaxes(0,1)
                view_mask = view_mask.swapaxes(1,2)
                visualize(view_img, view_mask)