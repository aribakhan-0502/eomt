# train_mvtec.py
import torch
import torch.nn as nn
from models.eomt_mrl import EoMT_MRL
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

class MVTecAnomalyDetection(pl.LightningModule):
    def __init__(self, eomt_mrl_model, learning_rate=1e-4):
        super().__init__()
        self.model = eomt_mrl_model
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Freeze backbone, train only MRL heads
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        
    def forward(self, x, rep_size=None):
        return self.model(x, rep_size)
    
    def training_step(self, batch, batch_idx):
        images, labels, masks = batch['image'], batch['label'], batch['mask']
        
        # Get predictions (use all scales during training)
        mask_logits, class_logits = self(images)
        
        # Simple anomaly detection loss
        # Use the largest scale for training
        largest_scale_logits = class_logits[-1].mean(dim=1)  # [B, num_classes+1]
        anomaly_scores = largest_scale_logits[:, 0]  # First class as anomaly score
        
        loss = self.criterion(anomaly_scores, labels.float())
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # Only train MRL components
        trainable_params = (
            list(self.model.class_head.parameters()) +
            list(self.model.mask_head.parameters())
        )
        
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        return optimizer

def setup_mvtec_training():
    # Load your EoMT configuration
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/dinov3/coco/panoptic/eomt_base_640_2x.yaml")
    
    # Load encoder
    from models.encoder import Encoder
    encoder = Encoder(**config.model.init_args.network.init_args.encoder.init_args)
    
    # Create EoMT-MRL for anomaly detection (2 classes: normal vs anomalous)
    eomt_mrl = EoMT_MRL(
        encoder=encoder,
        num_classes=1,  # Binary: normal vs anomaly
        num_q=100,
        nesting_list=[64, 128, 256, 512],
        efficient=True
    )
    
    # Load pre-trained weights (your .bin file)
    checkpoint_path = "path/to/your/eomt_base_640.bin"
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    eomt_mrl.load_state_dict(state_dict, strict=False)
    
    # Create Lightning module
    model = MVTecAnomalyDetection(eomt_mrl)
    
    return model

if __name__ == "__main__":
    model = setup_mvtec_training()
    
    # Setup data (you'll need to create MVTec data module)
    from data.mvtec import MVTecDataModule
    datamodule = MVTecDataModule(
        data_path="/path/to/mvtec/ad",
        category="bottle",
        batch_size=4
    )
    
    # Train
    trainer = Trainer(
        max_epochs=50,
        devices=1,
        accelerator="gpu",
        logger=WandbLogger(project="eomt-mrl-anomaly")
    )
    
    trainer.fit(model, datamodule)