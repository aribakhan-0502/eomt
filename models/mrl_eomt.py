# models/mrl_eomt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional
from models.eomt import EoMT
from models.scale_block import ScaleBlock

class MRL_EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        num_q: int,
        num_blocks: int = 4,
        masked_attn_enabled: bool = True,
        nesting_list: List[int] = None,
        efficient: bool = False,
    ):
        super().__init__()
        
        # EoMT components
        self.encoder = encoder
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled
        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))
        self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)

        # MRL components
        self.nesting_list = nesting_list or [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        self.efficient = efficient
        
        # MRL Class Head (for multi-scale anomaly classification)
        if self.efficient:
            self.class_head = MRL_Efficient_Linear_Layer(
                self.nesting_list, num_classes + 1, self.encoder.backbone.embed_dim
            )
        else:
            self.class_head = MRL_MultiHead_Linear_Layer(
                self.nesting_list, num_classes + 1, self.encoder.backbone.embed_dim
            )

        # MRL Mask Head (for multi-scale anomaly segmentation)
        self.mask_head = MRL_Mask_Head(
            self.nesting_list, self.encoder.backbone.embed_dim, efficient=efficient
        )

        # Upscaling for segmentation
        patch_size = encoder.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
        
        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.backbone.embed_dim) for _ in range(num_upscale)],
        )

    def _predict(self, x: torch.Tensor):
        q = x[:, : self.num_q, :]

        # Multi-scale class predictions
        class_logits_nested = self.class_head(q)

        # Multi-scale mask predictions
        x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
        )

        upscaled_features = self.upscale(x)
        mask_logits_nested = self.mask_head(q, upscaled_features)

        return mask_logits_nested, class_logits_nested

    # Keep existing EoMT attention methods...
    def _disable_attn_mask(self, attn_mask, prob):
        # Same as original EoMT
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self.encoder.backbone.num_prefix_tokens :
            ][random_queries] = True
        return attn_mask

    def _attn_mask(self, x: torch.Tensor, mask_logits: torch.Tensor, i: int):
        # Use the largest scale for attention masking
        largest_mask_logits = mask_logits[-1] if isinstance(mask_logits, list) else mask_logits
        
        attn_mask = torch.ones(
            x.shape[0],
            x.shape[1],
            x.shape[1],
            dtype=torch.bool,
            device=x.device,
        )
        interpolated = F.interpolate(
            largest_mask_logits,
            self.encoder.backbone.patch_embed.grid_size,
            mode="bilinear",
        )
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        attn_mask[
            :,
            : self.num_q,
            self.num_q + self.encoder.backbone.num_prefix_tokens :,
        ] = (
            interpolated > 0
        )
        attn_mask = self._disable_attn_mask(
            attn_mask,
            self.attn_mask_probs[
                i - len(self.encoder.backbone.blocks) + self.num_blocks
            ],
        )
        return attn_mask

    def _attn(
    self,
    module: nn.Module,
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    rope: Optional[torch.Tensor],):
        if rope is not None:
            if mask is not None:
                mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
            return module(x, mask, rope)[0]

        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

        return x

    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x)

        x = self.encoder.backbone.patch_embed(x)

        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        attn_mask = None
        mask_logits_per_layer_nested = []
        class_logits_per_layer_nested = []

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                mask_logits_nested, class_logits_nested = self._predict(self.encoder.backbone.norm(x))
                mask_logits_per_layer_nested.append(mask_logits_nested)
                class_logits_per_layer_nested.append(class_logits_nested)

                # Use largest scale for attention masking
                largest_mask_logits = mask_logits_nested[-1] if mask_logits_nested else None
                if largest_mask_logits is not None:
                    attn_mask = self._attn_mask(x, largest_mask_logits, i)

            # Original EoMT attention processing...
            if hasattr(block, "attn"):
                attn = block.attn
            else:
                attn = block.attention
                
            # Use original EoMT _attn method (you'll need to copy it)
            attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)
            
            if hasattr(block, "ls1"):
                x = x + block.ls1(attn_out)
            elif hasattr(block, "layer_scale1"):
                x = x + block.layer_scale1(attn_out)

            mlp_out = block.mlp(block.norm2(x))
            if hasattr(block, "ls2"):
                x = x + block.ls2(mlp_out)
            elif hasattr(block, "layer_scale2"):
                x = x + block.layer_scale2(mlp_out)

        # Final prediction
        mask_logits_nested, class_logits_nested = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer_nested.append(mask_logits_nested)
        class_logits_per_layer_nested.append(class_logits_nested)

        return mask_logits_per_layer_nested, class_logits_per_layer_nested


class MRL_MultiHead_Linear_Layer(nn.Module):
    def __init__(self, nesting_list: List[int], num_classes: int, embed_dim: int):
        super().__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes
        
        for i, num_feat in enumerate(self.nesting_list):
            setattr(self, f"nesting_classifier_{i}", nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        nesting_logits = ()
        for i, num_feat in enumerate(self.nesting_list):
            classifier = getattr(self, f"nesting_classifier_{i}")
            nesting_logits += (classifier(x),)
        return nesting_logits


class MRL_Efficient_Linear_Layer(nn.Module):
    def __init__(self, nesting_list: List[int], num_classes: int, embed_dim: int):
        super().__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        nesting_logits = ()
        for i, num_feat in enumerate(self.nesting_list):
            if self.classifier.bias is None:
                logit = torch.matmul(x, (self.classifier.weight[:, :num_feat]).t())
            else:
                logit = torch.matmul(x, (self.classifier.weight[:, :num_feat]).t()) + self.classifier.bias
            nesting_logits += (logit,)
        return nesting_logits


class MRL_Mask_Head(nn.Module):
    def __init__(self, nesting_list: List[int], embed_dim: int, efficient: bool = False):
        super().__init__()
        self.nesting_list = nesting_list
        self.efficient = efficient
        
        if efficient:
            self.mask_projector = nn.Linear(embed_dim, embed_dim)
        else:
            for i, num_feat in enumerate(nesting_list):
                setattr(self, f"mask_projector_{i}", nn.Linear(embed_dim, embed_dim))

    def forward(self, q, upscaled_features):
        nesting_mask_logits = ()
        
        for i, num_feat in enumerate(self.nesting_list):
            if self.efficient:
                mask_proj = self.mask_projector(q)
            else:
                mask_proj = getattr(self, f"mask_projector_{i}")(q)
                
            mask_logits = torch.einsum("bqc, bchw -> bqhw", mask_proj, upscaled_features)
            nesting_mask_logits += (mask_logits,)
            
        return nesting_mask_logits
