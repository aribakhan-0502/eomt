# datasets/transforms.py
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Union
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torchvision.transforms.v2 as T

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
        is_train: bool = True
    ):
        super().__init__()

        self.img_size = img_size
        self.color_jitter_enabled = color_jitter_enabled
        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0
        self.is_train = is_train

        # Create appropriate transforms based on mode
        if self.is_train:
            # Training transforms with augmentation
            self.transform_pipeline = T.Compose([
                T.Resize(img_size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(img_size, padding=4),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomGrayscale(p=0.1),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # Validation/Test transforms (minimal)
            self.transform_pipeline = T.Compose([
                T.Resize(img_size),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def forward(
        self, img: Tensor, target: dict[str, Union[Tensor, tv_tensors.TVTensor]]
    ) -> tuple[Tensor, dict[str, Union[Tensor, tv_tensors.TVTensor]]]:
        
        # Apply transforms to image
        img = self.transform_pipeline(img)
        
        # Resize masks to match image size (nearest neighbor for masks)
        if "masks" in target:
            target["masks"] = F.resize(
                target["masks"], 
                self.img_size, 
                interpolation=F.InterpolationMode.NEAREST
            )
        
        return img, target


def get_anomaly_transforms(img_size: tuple[int, int], is_train: bool = True):
    """Get transforms for anomaly detection"""
    return Transforms(img_size=img_size, is_train=is_train)
