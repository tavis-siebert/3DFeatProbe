import torch
import torch.nn as nn
from typing import Dict

from src.models.processors import BaseProcessor
from external.mum.mum import mum_vitl16

class MuMVisionTransformer(nn.Module):
    """
    Convenience wrapper to extract features from MuM Encoder.
    See https://github.com/davnords/mum# for their awesome work
    """
    def __init__(self, pretrained: bool=True, preprocess_images: bool=True, **kwargs):
        """
        Args:
            pretrained (bool): If true, uses checkpoint weights
            preprocess_images (bool): Whether to preprocess images inside the forward pass. Default = True
        """
        super().__init__()
        self.model = mum_vitl16(pretrained, **kwargs)
        self.patch_size = self.model.patch_size
        if preprocess_images:
            self.proecessor = BaseProcessor(patch_size=self.patch_size, normalize=True)

    def forward(self, images: torch.Tensor, unflatten_patches: bool=False) -> Dict[str, torch.Tensor]:
        """
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W).
            unflatten_patches (bool): Whether the patch embeddings are returned as a contiguous sequence 
                                    of patches or a 2D spatial map. Default is flattened (False).
        
        Returns:
            output_dict (Dict[str, torch.Tensor]): convenient views of last hidden states for downstream use
                "x_norm": the last hidden state in full (including CLS and possible register or other tokens)
                "x_norm_clstoken": the CLS token for classification or semantic tasks
                "x_norm_patchtokens": the patch embeddings
        """
        if self.proecessor:
            images = self.proecessor(images)
        
        num_patches_h, num_patches_w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size

        out = self.model.forward_features(images)

        x_norm = self.model.norm(out["x_prenorm"])

        patch_embeds = out["x_norm_patchtokens"]
        if unflatten_patches:
            patch_embeds = patch_embeds.unflatten(1, (num_patches_h, num_patches_w))

        return {
            "x_norm": x_norm,
            "x_norm_clstoken": out["x_norm_cls_token"],
            "x_norm_patchtokens": patch_embeds
        }
            