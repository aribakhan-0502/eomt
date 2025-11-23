# datasets/transforms.py
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Union
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import TVTensor
from torchvision.transforms.v2._utils import wrap

class Transforms(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        color_jitter_enabled: bool = True,
        scale_range: tuple[float, float] = (0.1, 2.0),
        max_brightness_delta: int = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: int = 18,
        is_train: bool = True  # Add this parameter for train vs val
    ):
        super().__init__()

        self.img_size = img_size
        self.color_jitter_enabled = color_jitter_enabled
        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0
        self.is_train = is_train

        # Only use augmentation transforms for training
        if self.is_train:
            self.random_horizontal_flip = T.RandomHorizontalFlip()
            self.scale_jitter = T.ScaleJitter(target_size=img_size, scale_range=scale_range)
            self.random_crop = T.RandomCrop(img_size)
            self.color_jitter_transform = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
            self.random_grayscale = T.RandomGrayscale(p=0.1)
            self.gaussian_blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        else:
            # For validation/test, just use resize
            self.resize = T.Resize(img_size)

        # Common transforms for both train and val
        self.to_dtype = T.ToDtype(torch.float32, scale=True)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _random_factor(self, factor: float, center: float = 1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def _brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img, self._random_factor(self.max_brightness_factor)
            )
        return img

    def _contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(img, self._random_factor(self.max_contrast_factor))
        return img

    def _saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img, self._random_factor(self.max_saturation_factor)
            )
        if torch.rand(()) < 0.5:
            img = F.adjust_hue(img, self._random_factor(self.max_hue_delta, center=0.0))
        return img

    def color_jitter(self, img):
        if not self.color_jitter_enabled or not self.is_train:
            return img

        img = self._brightness(img)

        if torch.rand(()) < 0.5:
            img = self._contrast(img)
            img = self._saturation_and_hue(img)
        else:
            img = self._saturation_and_hue(img)
            img = self._contrast(img)

        return img

    def pad(
        self, img: Tensor, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding)
        target["masks"] = F.pad(target["masks"], padding)

        return img, target

    def _filter(self, target: dict[str, Union[Tensor, TVTensor]], keep: Tensor) -> dict:
        return {k: wrap(v[keep], like=v) for k, v in target.items()}

    def forward(
        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        
        if not self.is_train:
            # Validation/Test mode: simple resize and normalize
            img = self.resize(img)
            target["masks"] = F.resize(target["masks"], self.img_size, interpolation=F.InterpolationMode.NEAREST)
        else:
            # Training mode: full augmentation pipeline
            img_orig, target_orig = img, target
            target = self._filter(target, ~target["is_crowd"])

            img = self.color_jitter(img)
            img, target = self.random_horizontal_flip(img, target)
            img, target = self.scale_jitter(img, target)
            img, target = self.pad(img, target)
            img, target = self.random_crop(img, target)

            # Apply additional augmentations
            img = self.color_jitter_transform(img)
            img = self.random_grayscale(img)
            img = self.gaussian_blur(img)

            valid = target["masks"].flatten(1).any(1)
            if not valid.any():
                return self(img_orig, target_orig)

            target = self._filter(target, valid)

        # Common processing for both train and val
        img = self.to_dtype(img)
        img = self.normalize(img)

        return img, target


def get_anomaly_transforms(img_size: tuple[int, int], is_train: bool = True):
    """Get transforms for anomaly detection - wrapper for the Transforms class"""
    return Transforms(img_size=img_size, is_train=is_train)
