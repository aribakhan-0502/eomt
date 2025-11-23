# datasets/transforms.py
import torch
from torchvision.transforms.v2 import (
    Compose, Resize, RandomHorizontalFlip, RandomCrop, ColorJitter, 
    RandomGrayscale, GaussianBlur, ToDtype, Normalize
)

def get_anomaly_transforms(img_size: tuple[int, int], is_train: bool = True):
    """Get transforms for anomaly detection"""
    
    if is_train:
        return Compose([
            Resize(img_size),
            RandomHorizontalFlip(p=0.5),
            RandomCrop(img_size, padding=4),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            RandomGrayscale(p=0.1),
            GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return Compose([
            Resize(img_size),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
