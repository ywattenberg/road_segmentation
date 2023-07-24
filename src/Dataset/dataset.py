from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os
import numpy as np

# [0.48015234 0.47847242 0.44929579 1.        ]
# [0.2111749  0.19588756 0.19973656 0.        ]


class BaseDataset(Dataset):
    def __init__(self, augment_images=False, normalize=False) -> None:
        self.length = 0
        self.augment_images = augment_images
        self.normalize = normalize
        self.augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90, fill=(0)),
                transforms.RandomCrop(400, fill=(0)),
            ]
        )
        self.color_augment = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        self.norm = transforms.Normalize(
            mean=[0.48, 0.478, 0.449, 0.5], std=[0.211, 0.196, 0.200, 0.5]
        )

        self.pad = transforms.Pad(56, fill=(0))
        # self.color_augment = transforms.v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        super().__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raise NotImplementedError

    def augment_image(self, *args):
        aug = self.augment(torch.cat(args))
        if len(args) == 3:
            return aug[0:3, :, :], aug[3, :, :], aug[4, :, :]
        else:
            return aug[0:3, :, :], aug[3, :, :]

    def resize(self, *args):
        return [self.pad(arg) for arg in args]


class ETHDataset(BaseDataset):
    def __init__(
        self,
        image_path,
        mask_path,
        skel_path=None,
        augment_images=False,
        normalize=False,
        submission=False,
    ) -> None:
        super().__init__()
        self.augment_images = augment_images
        self.image_path = image_path
        self.mask_path = mask_path
        self.image_list = os.listdir(self.image_path)
        self.length = len(self.image_list)
        self.skel_path = skel_path
        self.normalize = normalize
        self.submission = submission

    def __getitem__(self, index):
        # Image should be in RGBA format
        image = Image.open(os.path.join(self.image_path, self.image_list[index]))
        image = image.convert("RGB")
        image = transforms.ToTensor()(image)

        if self.normalize:
            image = self.norm(image)
        if self.submission:
            return self.resize(image)[0]

        mask = Image.open(os.path.join(self.mask_path, self.image_list[index]))
        mask = transforms.ToTensor()(mask)
        if self.skel_path is not None:
            skeleton = Image.open(os.path.join(self.skel_path, self.image_list[index]))
            skeleton = transforms.ToTensor()(skeleton)

            augmented_stack = (
                self.augment_image(image, mask, skeleton)
                if self.augment
                else [image, mask, skeleton]
            )
        else:
            augmented_stack = (
                self.augment_image(image, mask) if self.augment else [image, mask]
            )
        return self.resize(*augmented_stack)


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
        image = Image.open(os.path.join(self.image_path, self.image_list[index]))
        # .convert("RGBA")
        mask = Image.open(os.path.join(self.mask_path, self.image_list[index])[:-1])
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        resized = transforms.RandomCrop(400, 0)(torch.cat([image, mask]))
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
        image = Image.open(
            os.path.join(self.image_path, self.image_list[index])
        ).convert("RGB")
        image = transforms.ToTensor()(image)

        mask = Image.open(os.path.join(self.mask_path, self.image_list[index]))
        mask = transforms.ToTensor()(mask)
        if self.skel_path is not None:
            skeleton = Image.open(os.path.join(self.skel_path, self.image_list[index]))
            skeleton = transforms.ToTensor()(skeleton)

            augmented_stack = (
                self.augment_image(image, mask, skeleton)
                if self.augment
                else [image, mask, skeleton]
            )
        else:
            augmented_stack = (
                self.augment_image(image, mask) if self.augment else [image, mask]
            )
        return self.resize(*augmented_stack)
        # return image, mask, skeleton
