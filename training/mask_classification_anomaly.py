# training/mask_classification_anomaly.py - NEW FILE
import torch
import torch.nn as nn
from training.lightning_module import LightningModule
from training.mask_classification_loss import MaskClassificationCriterion

class MaskClassificationAnomaly(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool = False,
        attn_mask_annealing_start_steps: list[int] = None,
        attn_mask_annealing_end_steps: list[int] = None,
        lr: float = 2e-4,
        llrd: float = 1.0,
        llrd_l2_enabled: bool = False,
        lr_mult: float = 1.0,
        weight_decay: float = 1e-4,
        poly_power: float = 1.0,
        warmup_steps: tuple[int, int] = (2000, 3000),
        ckpt_path=None,
        delta_weights=False,
        load_ckpt_class_head=True,
    ):
        # For anomaly detection, we use binary classification
        actual_num_classes = 1  # Normal (0) vs Anomalous (1)
        
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=actual_num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )
        
        # Use the existing loss criterion
        self.criterion = MaskClassificationCriterion(
            num_classes=actual_num_classes + 1,  # +1 for "no object"
            eos_coef=0.1,
            losses=["labels", "masks"],
        )
        
        # For anomaly detection metrics
        self.anomaly_threshold = 0.5

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return super().configure_optimizers()