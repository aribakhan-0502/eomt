# training/mrl_anomaly_detection.py
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from torchmetrics.detection import MeanAveragePrecision

from .mrl_anomaly_loss import MRL_Anomaly_Loss
from .lightning_module import LightningModule


class MRL_AnomalyDetection(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int = 2,
        nesting_list: list = None,
        relative_importance: list = None,
        **kwargs
    ):
        super().__init__(network=network, img_size=img_size, num_classes=num_classes, **kwargs)
        
        self.nesting_list = nesting_list or [64, 128, 256, 512, 1024]
        self.relative_importance = relative_importance or [1.0] * len(self.nesting_list)
        
        # Replace criterion with MRL anomaly loss
        self.criterion = MRL_Anomaly_Loss(
            num_points=12544,  # Adjust based on your needs
            oversample_ratio=0.75,
            importance_sample_ratio=0.75,
            mask_coefficient=1.0,
            dice_coefficient=1.0,
            class_coefficient=1.0,
            num_labels=num_classes,
            no_object_coefficient=0.1,
            nesting_list=self.nesting_list,
            relative_importance=self.relative_importance,
        )
        
        # Metrics for anomaly detection
        self.train_accuracy = Accuracy(task="binary", num_classes=2)
        self.val_accuracy = Accuracy(task="binary", num_classes=2)
        self.test_accuracy = Accuracy(task="binary", num_classes=2)
        
        self.val_precision = Precision(task="binary", num_classes=2)
        self.val_recall = Recall(task="binary", num_classes=2)
        self.val_f1 = F1Score(task="binary", num_classes=2)
        self.val_auroc = AUROC(task="binary", num_classes=2)
        
        # For segmentation metrics - COMMENTED OUT TEMPORARILY
        # self.val_map = MeanAveragePrecision(iou_type="segm")

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        
        mask_logits_per_block_nested, class_logits_per_block_nested = self(imgs)

        losses_all_blocks = {}
        for block_idx, (mask_logits_block, class_logits_block) in enumerate(
            zip(mask_logits_per_block_nested, class_logits_per_block_nested)
        ):
            losses = self.criterion(
                masks_queries_logits_nested=[mask_logits_block],
                class_queries_logits_nested=[class_logits_block],
                targets=targets,
            )
            
            block_postfix = self.block_postfix(block_idx)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        # Calculate image-level anomaly accuracy
        anomaly_scores = self._get_anomaly_scores(class_logits_per_block_nested[-1][-1])
        anomaly_labels = torch.tensor([t["anomaly_label"] for t in targets]).to(imgs.device)
        self.train_accuracy(anomaly_scores, anomaly_labels)

        return self.criterion.loss_total(losses_all_blocks, self.log)

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "test")

    def _shared_eval_step(self, batch, batch_idx, prefix):
        imgs, targets = batch
        
        with torch.no_grad():
            mask_logits_per_block_nested, class_logits_per_block_nested = self(imgs)
            
            # Use the final block and largest nesting scale for evaluation
            final_mask_logits = mask_logits_per_block_nested[-1][-1]
            final_class_logits = class_logits_per_block_nested[-1][-1]
            
            # Image-level anomaly detection
            anomaly_scores = self._get_anomaly_scores(final_class_logits)
            anomaly_labels = torch.tensor([t["anomaly_label"] for t in targets]).to(imgs.device)
            
            # Update metrics
            if prefix == "val":
                self.val_accuracy(anomaly_scores, anomaly_labels)
                self.val_precision(anomaly_scores, anomaly_labels)
                self.val_recall(anomaly_scores, anomaly_labels)
                self.val_f1(anomaly_scores, anomaly_labels)
                self.val_auroc(anomaly_scores, anomaly_labels)
                
                # Segmentation metrics for anomalous regions - COMMENTED OUT TEMPORARILY
                # pred_masks = self._get_predicted_masks(final_mask_logits, final_class_logits)
                # self.val_map.update(pred_masks, targets)
            
            elif prefix == "test":
                self.test_accuracy(anomaly_scores, anomaly_labels)

    def _get_anomaly_scores(self, class_logits):
        """Get anomaly scores from class logits"""
        # class_logits shape: [batch_size, num_queries, num_classes+1]
        # Use the "anomalous" class probability (class 1)
        probs = torch.softmax(class_logits, dim=-1)  # [batch_size, num_queries, num_classes+1]
        anomaly_probs = probs[:, :, 1]  # Probability of being anomalous
        
        # Take max probability across queries
        anomaly_scores, _ = torch.max(anomaly_probs, dim=1)
        
        # Binary predictions
        return (anomaly_scores > 0.5).long()

    def _get_predicted_masks(self, mask_logits, class_logits, mask_threshold=0.5):
        """Convert model outputs to detection format for metrics"""
        batch_size = mask_logits.shape[0]
        predictions = []
    
        for i in range(batch_size):
            # Get masks and class predictions
            masks = torch.sigmoid(mask_logits[i])  # [num_queries, H, W]
            class_probs = torch.softmax(class_logits[i], dim=-1)  # [num_queries, num_classes+1]
        
            # Filter out background and low-confidence predictions
            scores, labels = torch.max(class_probs[:, :-1], dim=-1)  # Exclude background
            keep = scores > 0.1  # Confidence threshold
        
            if keep.any():
                pred_masks = masks[keep]
                pred_scores = scores[keep]
                pred_labels = labels[keep]
            
                # Convert to binary masks and ensure uint8 dtype
                binary_masks = (pred_masks > mask_threshold)
                binary_masks = binary_masks.to(torch.uint8)
            
                prediction = {
                    "masks": binary_masks,
                    "scores": pred_scores,
                    "labels": pred_labels,
                }
            else:
                prediction = {
                    "masks": torch.zeros(0, *mask_logits.shape[-2:], dtype=torch.uint8),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.long),
                }
        
            predictions.append(prediction)
    
        return predictions

    def on_validation_epoch_end(self):
        # Log anomaly detection metrics
        self.log("metrics/val_accuracy", self.val_accuracy.compute(), prog_bar=True)
        self.log("metrics/val_precision", self.val_precision.compute())
        self.log("metrics/val_recall", self.val_recall.compute())
        self.log("metrics/val_f1", self.val_f1.compute())
        self.log("metrics/val_auroc", self.val_auroc.compute())
        
        # Log segmentation metrics - COMMENTED OUT TEMPORARILY
        # map_results = self.val_map.compute()
        # self.log("metrics/val_map", map_results["map"])
        # self.log("metrics/val_map_50", map_results["map_50"])
        
        # Reset metrics
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        # self.val_map.reset()  # COMMENTED OUT

    def on_test_epoch_end(self):
        self.log("metrics/test_accuracy", self.test_accuracy.compute())
        self.test_accuracy.reset()

    def block_postfix(self, block_idx):
        if not self.network.masked_attn_enabled:
            return ""
        return (
            f"_block_{-self.network.num_blocks + block_idx + 1}"
            if block_idx != self.network.num_blocks
            else ""
        )
