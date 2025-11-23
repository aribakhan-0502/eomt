# datasets/mvtec_data_module.py
import torch
from pathlib import Path
from typing import Optional, List
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .mvtec_anomaly import MVTecAnomaly
from .transforms import get_anomaly_transforms


class MVTecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        category: str = "bottle",
        img_size: tuple[int, int] = (640, 640),
        batch_size: int = 8,
        num_workers: int = 8,
        train_transforms=None,
        val_transforms=None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.category = category
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_transforms = train_transforms or get_anomaly_transforms(
            img_size, is_train=True
        )
        self.val_transforms = val_transforms or get_anomaly_transforms(
            img_size, is_train=False
        )
        
        self.num_classes = 2  # Normal (0) and Anomalous (1)

    def setup(self, stage: Optional[str] = None):
        # Verify dataset path exists
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset path {self.data_path} does not exist. "
                f"Please check your Google Drive mounting and path."
            )
        
        category_path = self.data_path / self.category
        if not category_path.exists():
            raise FileNotFoundError(
                f"Category {self.category} not found in {self.data_path}. "
                f"Available categories: {[d.name for d in self.data_path.iterdir() if d.is_dir()]}"
            )
        
        print(f"✅ Loading MVTec AD category: {self.category}")
        print(f"📁 Dataset path: {self.data_path}")
        
        if stage == "fit" or stage is None:
            self.train_dataset = MVTecAnomaly(
                data_path=self.data_path,
                category=self.category,
                split="train",
                img_size=self.img_size,
                transforms=self.train_transforms,
            )
            
            self.val_dataset = MVTecAnomaly(
                data_path=self.data_path,
                category=self.category,
                split="test",
                img_size=self.img_size,
                transforms=self.val_transforms,
            )
            
            print(f"📊 Training samples: {len(self.train_dataset)}")
            print(f"📊 Validation samples: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            self.test_dataset = MVTecAnomaly(
                data_path=self.data_path,
                category=self.category,
                split="test",
                img_size=self.img_size,
                transforms=self.val_transforms,
            )
            print(f"📊 Test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable-sized masks"""
        images = torch.stack([item[0] for item in batch])  # This line needs torch
        targets = []
        
        for _, target in batch:
            # Ensure masks have consistent shape
            if target["masks"].dim() == 2:  # Single mask
                target["masks"] = target["masks"].unsqueeze(0)
            targets.append(target)
            
        return images, targets

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer batch to device"""
        images, targets = batch
        images = images.to(device)
        
        for target in targets:
            for key in target:
                if isinstance(target[key], torch.Tensor):
                    target[key] = target[key].to(device)
                    
        return images, targets
