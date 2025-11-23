# train_simple.py
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from datasets.mvtec_data_module import MVTecDataModule
from training.mrl_anomaly_detection import MRL_AnomalyDetection
from models.mrl_eomt import MRL_EoMT
from models.vit import ViT

def main():
    # Configuration
    data_path = "/content/drive/MyDrive/MS_Anomaly_Detection/01_datasets/mvtec_ad"
    category = "bottle"
    img_size = (640, 640)
    batch_size = 8
    max_epochs = 50
    
    # Data module
    data_module = MVTecDataModule(
        data_path=data_path,
        category=category,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=4,
    )
    
    # Model
    encoder = ViT(
        img_size=img_size,
        backbone_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
    )
    
    network = MRL_EoMT(
        encoder=encoder,
        num_classes=2,
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
        attn_mask_annealing_enabled=True,
        attn_mask_annealing_start_steps=[0, 5000, 10000],
        attn_mask_annealing_end_steps=[2000, 7000, 15000],
        llrd=0.95,
        llrd_l2_enabled=False,
        lr_mult=1.0,
        weight_decay=0.01,
        poly_power=0.9,
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
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stopping],
        logger=pl.loggers.WandbLogger(project="mrl_eomt_mvtec", name=f"{category}_anomaly"),
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Test
    trainer.test(model, data_module)

if __name__ == "__main__":
    main()