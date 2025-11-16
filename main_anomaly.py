import logging
import torch
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from training.mrl_anomaly_detection import MRL_AnomalyDetection
from datasets.mvtec_data_module import MVTecDataModule

def main():
    logging.getLogger().setLevel(logging.INFO)
    torch.set_float32_matmul_precision("medium")
    
    LightningCLI(
        MRL_AnomalyDetection,
        MVTecDataModule,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "devices": 1,
        },
    )

if __name__ == "__main__":
    main()