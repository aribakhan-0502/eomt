# data/mvtec.py
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .mvtec_dataset import MVTecAnomalyDataset

class MVTecDataModule(pl.LightningDataModule):
    def __init__(self, data_path, category="bottle", batch_size=4, image_size=640):
        super().__init__()
        self.data_path = data_path
        self.category = category
        self.batch_size = batch_size
        self.image_size = image_size
    
    def setup(self, stage=None):
        self.train_dataset = MVTecAnomalyDataset(
            self.data_path, self.category, "train", self.image_size
        )
        self.val_dataset = MVTecAnomalyDataset(
            self.data_path, self.category, "test", self.image_size
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)