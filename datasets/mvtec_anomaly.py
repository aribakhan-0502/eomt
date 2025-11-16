# datasets/mvtec_anomaly.py
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from PIL import Image
import numpy as np

from .dataset import Dataset


class MVTecAnomaly(Dataset):
    def __init__(
        self,
        data_path: Path,
        category: str = "bottle",
        split: str = "train",
        img_size: Tuple[int, int] = (640, 640),
        transforms: Optional[Any] = None,
        **kwargs
    ):
        self.data_path = Path(data_path)
        self.category = category
        self.split = split
        self.img_size = img_size
        
        # For anomaly detection, we have 2 classes: normal (0) and anomalous (1)
        self.num_classes = 2
        
        # Build the dataset structure
        self.imgs = []
        self.labels = []  # 0 for normal, 1 for anomalous
        self.mask_paths = []  # Path to ground truth masks for anomalous samples
        
        self._build_dataset()
        
        super().__init__(
            zip_path=self.data_path,  # Not using zip files, using directory structure
            img_suffix=".png",
            target_parser=self._target_parser,
            check_empty_targets=False,
            transforms=transforms,
            **kwargs
        )

    def _build_dataset(self):
        """Build the dataset from MVTec AD directory structure"""
        if self.split == "train":
            # Training only contains normal samples
            train_good_path = self.data_path / self.category / "train" / "good"
            if train_good_path.exists():
                for img_file in sorted(train_good_path.glob("*.png")):
                    self.imgs.append(str(img_file))
                    self.labels.append(0)  # Normal
                    self.mask_paths.append(None)  # No masks for normal samples
                    
        elif self.split == "test":
            # Test contains both normal and anomalous samples
            test_good_path = self.data_path / self.category / "test" / "good"
            if test_good_path.exists():
                for img_file in sorted(test_good_path.glob("*.png")):
                    self.imgs.append(str(img_file))
                    self.labels.append(0)  # Normal
                    self.mask_paths.append(None)
            
            # Add anomalous samples
            test_path = self.data_path / self.category / "test"
            for defect_type in test_path.iterdir():
                if defect_type.name == "good":
                    continue
                    
                if defect_type.is_dir():
                    for img_file in sorted(defect_type.glob("*.png")):
                        self.imgs.append(str(img_file))
                        self.labels.append(1)  # Anomalous
                        
                        # Find corresponding mask
                        mask_file = self._find_mask_file(img_file, defect_type.name)
                        self.mask_paths.append(mask_file)

    def _find_mask_file(self, img_file: Path, defect_type: str) -> Optional[Path]:
        """Find the corresponding ground truth mask for an anomalous image"""
        gt_path = self.data_path / self.category / "ground_truth" / defect_type
        if not gt_path.exists():
            return None
            
        img_name = img_file.stem
        mask_file = gt_path / f"{img_name}_mask.png"
        
        if mask_file.exists():
            return mask_file
        return None

    def _target_parser(self, target=None, target_instance=None, **kwargs):
        """
        Parse targets for anomaly detection.
        Returns masks, labels, and is_crowd for each instance.
        """
        masks = []
        labels = []
        is_crowd = []
        
        idx = kwargs.get('index', 0)
        
        if self.labels[idx] == 1 and self.mask_paths[idx] is not None:
            # Anomalous sample with mask
            try:
                mask = Image.open(self.mask_paths[idx]).convert('L')
                mask_tensor = tv_tensors.Mask(mask)
                
                # Resize mask to match image size if needed
                if mask_tensor.shape[-2:] != self.img_size:
                    mask_tensor = F.resize(
                        mask_tensor,
                        list(self.img_size),
                        interpolation=F.InterpolationMode.NEAREST,
                    )
                
                # Convert to binary mask (anomalous regions)
                mask_binary = (mask_tensor > 0).float()
                
                if mask_binary.sum() > 0:  # Only add if there are anomalous pixels
                    masks.append(mask_binary)
                    labels.append(1)  # Anomalous class
                    is_crowd.append(False)
                    
            except Exception as e:
                print(f"Error loading mask {self.mask_paths[idx]}: {e}")
        
        # If no masks found (normal sample or mask loading failed), create empty mask
        if not masks:
            empty_mask = torch.zeros(self.img_size, dtype=torch.float32)
            masks.append(empty_mask)
            labels.append(0)  # Normal class
            is_crowd.append(False)
        
        return masks, labels, is_crowd

    def __getitem__(self, index: int):
        """Get item with proper index handling"""
        # Load image
        img_path = self.imgs[index]
        img = tv_tensors.Image(Image.open(img_path).convert("RGB"))
        
        # Resize image if needed
        if img.shape[-2:] != self.img_size:
            img = F.resize(img, list(self.img_size))
        
        # Parse target
        masks, labels, is_crowd = self._target_parser(index=index)
        
        target = {
            "masks": tv_tensors.Mask(torch.stack(masks) if masks else torch.zeros(0, *self.img_size)),
            "labels": torch.tensor(labels),
            "is_crowd": torch.tensor(is_crowd),
            "anomaly_label": torch.tensor(self.labels[index]),  # Overall image label
            "image_path": img_path
        }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)

    def close(self):
        """Override close method for directory-based dataset"""
        pass

    def _load_zips(self):
        """Override zip loading for directory-based dataset"""
        return None, None, None