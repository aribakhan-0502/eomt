# train_anomaly.py
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from datasets.mvtec_data_module import MVTecDataModule
from training.mrl_anomaly_detection import MRL_AnomalyDetection
from models.mrl_eomt import MRL_EoMT
from models.vit import ViT

def train_anomaly_detection(
    data_path: str,
    category: str = "bottle",
    img_size: tuple = (640, 640),
    batch_size: int = 8,
    num_workers: int = 8,
    max_epochs: int = 50,
    devices: int = 1,
):
    """Train MRL-EoMT for anomaly detection on MVTec AD"""
    
    # Data module
    data_module = MVTecDataModule(
        data_path=data_path,
        category=category,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Model
    encoder = ViT(
        img_size=img_size,
        backbone_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
    )
    
    network = MRL_EoMT(
        encoder=encoder,
        num_classes=2,  # Normal vs Anomalous
        num_q=100,
        num_blocks=3,
        nesting_list=[64, 128, 256, 512, 1024],
        efficient=False,
    )
    
    model = MRL_AnomalyDetection(
        network=network,
        img_size=img_size,
        num_classes=2,
        nesting_list=[64, 128, 256, 512, 1024],
        relative_importance=[0.1, 0.2, 0.3, 0.2, 0.2],
        lr=1e-4,
        warmup_steps=[1000, 2000],
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/val_accuracy",
        mode="max",
        save_top_k=3,
        filename=f"{category}-" + "{epoch:02d}-{val_accuracy:.3f}",
    )
    
    early_stopping = EarlyStopping(
        monitor="metrics/val_accuracy",
        patience=10,
        mode="max",
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stopping],
        logger=pl.loggers.WandbLogger(project="mrl_eomt_mvtec", name=f"{category}_anomaly"),
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Test
    trainer.test(model, data_module)
    
    return trainer, model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to MVTec AD dataset")
    parser.add_argument("--category", type=str, default="bottle",
                       help="MVTec category to train on")
    parser.add_argument("--img_size", type=int, nargs=2, default=[640, 640],
                       help="Image size (height width)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--devices", type=int, default=1)
    
    args = parser.parse_args()
    
    train_anomaly_detection(
        data_path=args.data_path,
        category=args.category,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        devices=args.devices,
    )