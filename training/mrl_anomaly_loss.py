# training/mrl_anomaly_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from training.mask_classification_loss import MaskClassificationLoss

class MRL_Anomaly_Loss(nn.Module):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        nesting_list: List[int] = None,
        relative_importance: List[float] = None,
    ):
        super().__init__()
        
        self.nesting_list = nesting_list or [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        self.relative_importance = relative_importance or [1.0] * len(self.nesting_list)
        
        # Base loss from MaskClassificationLoss
        self.base_loss = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_labels,
            no_object_coefficient=no_object_coefficient,
        )

    def forward(self, masks_queries_logits_nested, class_queries_logits_nested, targets):
        """
        masks_queries_logits_nested: list of lists [blocks][nesting_scales]
        class_queries_logits_nested: list of lists [blocks][nesting_scales]
        """
        losses_all_blocks_nested = {}
        
        # Process each block
        for block_idx, (mask_logits_block, class_logits_block) in enumerate(
            zip(masks_queries_logits_nested, class_queries_logits_nested)
        ):
            # Process each nesting scale
            for scale_idx, (mask_logits, class_logits) in enumerate(
                zip(mask_logits_block, class_logits_block)
            ):
                scale_losses = self.base_loss(
                    masks_queries_logits=mask_logits,
                    class_queries_logits=class_logits,
                    targets=targets,
                )
                
                # Apply relative importance and nesting scale identifier
                scale_postfix = f"_nesting_{self.nesting_list[scale_idx]}"
                block_postfix = f"_block_{block_idx}" if len(masks_queries_logits_nested) > 1 else ""
                
                for key, value in scale_losses.items():
                    weighted_loss = value * self.relative_importance[scale_idx]
                    losses_all_blocks_nested[f"{key}{scale_postfix}{block_postfix}"] = weighted_loss

        return losses_all_blocks_nested

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        loss_total = None
        
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            # Determine loss type and apply appropriate coefficient
            if "mask" in loss_key:
                weighted_loss = loss * self.base_loss.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.base_loss.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.base_loss.class_coefficient
            else:
                weighted_loss = loss

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)
        return loss_total
