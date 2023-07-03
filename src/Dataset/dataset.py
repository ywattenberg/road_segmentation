from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, augment_images=False) -> None:
        self.length = 0
        self.augment_images = augment_images
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90, fill=(0)),
            transforms.RandomCrop(400, fill=(0)),
        ])
        # self.color_augment = transforms.v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        super().__init__()

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        raise NotImplementedError

    def augment_image(self, image, mask):
        torch.stack([image, mask])
        aug = self.augment(torch.stack([image, mask]))
        return aug[0], aug[1][0, :, :]        

    
class ETHDataset(BaseDataset):
    def __init__(self, image_path, mask_path, augment_images=False) -> None:
        super().__init__()
        self.augment_images = augment_images
        self.image_path = image_path
        self.mask_path = mask_path
        self.image_list = os.listdir(self.image_path)
        self.mask_list = os.listdir(self.mask_path)
        self.length = len(self.image_list)
    
    def __getitem__(self, index):
        # Image should be in RGBA format
        image = Image.open(os.path.join(self.image_path, self.image_list[index]))
        mask = Image.open(os.path.join(self.mask_path, self.mask_list[index]))
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        if self.augment_images:
            mask = torch.cat([mask, torch.zeros(3, 400, 400)])
            augmented_stack = self.augment_image(image, mask)
            image = augmented_stack[0]
            mask = augmented_stack[1]
        return image, mask

    

    