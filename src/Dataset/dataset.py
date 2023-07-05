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

        self.pad = transforms.Pad(56, fill=(0))
        # self.color_augment = transforms.v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        super().__init__()

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        raise NotImplementedError

    def augment_image(self, image, mask):
        aug = self.augment(torch.cat([image, mask]))
        return aug[0:4, :, :], aug[4, :, :]        
    
    def resize(self, image, mask):
        return self.pad(image), self.pad(mask)

    
class ETHDataset(BaseDataset):
    def __init__(self, image_path, mask_path, augment_images=False) -> None:
        super().__init__()
        self.augment_images = augment_images
        self.image_path = image_path
        self.mask_path = mask_path
        self.image_list = os.listdir(self.image_path)
        self.length = len(self.image_list)
    
    def __getitem__(self, index):
        # Image should be in RGBA format
        image = Image.open(os.path.join(self.image_path, self.image_list[index]))
        mask = Image.open(os.path.join(self.mask_path, self.image_list[index]))
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        if self.augment_images:
            augmented_stack = self.augment_image(image, mask)
            image = augmented_stack[0]
            mask = augmented_stack[1]
        return self.resize(image, mask)

class MassachusettsDataset(BaseDataset):
    def __init__(self, image_path, mask_path, augment_images=False) -> None:
        super().__init__(augment_images)
        assert os.path.exists(image_path), "Image path does not exist"
        assert os.path.exists(mask_path), "Mask path does not exist"
        self.image_path = image_path
        self.mask_path = mask_path
        self.image_list = os.listdir(self.image_path)
        self.length = len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_path, self.image_list[index])).convert("RGBA")
        mask = Image.open(os.path.join(self.mask_path, self.image_list[index])[:-1])
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        resized  = transforms.RandomCrop(400, 0)(torch.cat([image, mask]))
        image = resized[0:4, :, :]
        mask = resized[4, :, :]
    
        if self.augment_images:
            
            augmented_stack = self.augment_image(image, mask.unsqueeze(0))
            image = augmented_stack[0]
            mask = augmented_stack[1]
            return self.resize(image, mask)
        else:
            return self.resize(image, mask)

    

class GMapsDataset(ETHDataset):
    def __getitem__(self, index):
        # Image should be in RGBA format
        image = Image.open(os.path.join(self.image_path, self.image_list[index])).convert("RGBA")
        mask = Image.open(os.path.join(self.mask_path, self.image_list[index]))
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        if self.augment_images:
            augmented_stack = self.augment_image(image, mask)
            image = augmented_stack[0]
            mask = augmented_stack[1]
        return self.resize(image, mask)