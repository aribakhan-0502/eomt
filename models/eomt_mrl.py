# models/eomt_mrl.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .eomt import EoMT
from .scale_block import ScaleBlock

class MRL_Linear_Layer(nn.Module):
    def __init__(self, nesting_list: list, num_classes=1, efficient=True):
        super(MRL_Linear_Layer, self).__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes
        self.efficient = efficient
        
        if self.efficient:
            self.classifier = nn.Linear(nesting_list[-1], self.num_classes)
        else:
            self.classifiers = nn.ModuleList([
                nn.Linear(num_feat, self.num_classes) for num_feat in nesting_list
            ])
    
    def forward(self, x, rep_size=None):
        if rep_size is not None:
            if self.efficient:
                if self.classifier.bias is None:
                    return torch.matmul(x[:, :rep_size], self.classifier.weight[:, :rep_size].t())
                else:
                    return torch.matmul(x[:, :rep_size], self.classifier.weight[:, :rep_size].t()) + self.classifier.bias
            else:
                idx = self.nesting_list.index(rep_size)
                return self.classifiers[idx](x[:, :rep_size])
        else:
            outputs = []
            for i, num_feat in enumerate(self.nesting_list):
                if self.efficient:
                    if self.classifier.bias is None:
                        output = torch.matmul(x[:, :num_feat], self.classifier.weight[:, :num_feat].t())
                    else:
                        output = torch.matmul(x[:, :num_feat], self.classifier.weight[:, :num_feat].t()) + self.classifier.bias
                else:
                    output = self.classifiers[i](x[:, :num_feat])
                outputs.append(output)
            return outputs

class EoMT_MRL(EoMT):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
        nesting_list=[64, 128, 256, 512],
        efficient=True
    ):
        super().__init__(encoder, num_classes, num_q, num_blocks, masked_attn_enabled)
        
        # MRL integration
        self.nesting_list = nesting_list
        self.efficient = efficient
        
        # Replace the class_head with MRL version
        self.class_head = MRL_Linear_Layer(
            nesting_list=nesting_list,
            num_classes=num_classes + 1,  # +1 for "no object"
            efficient=efficient
        )
        
        print(f"✅ EoMT-MRL initialized with scales: {nesting_list}")

    def forward(self, x: torch.Tensor, rep_size=None):
        # Use original EoMT forward pass
        mask_logits_per_layer, class_logits_per_layer = super().forward(x)
        
        # For now, we'll apply MRL at the class prediction level
        # In a more advanced version, we'd modify the internal _predict method
        return mask_logits_per_layer, class_logits_per_layer