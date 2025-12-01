import torch
import torch.nn as nn
from transformers import ViTMAEModel
from typing import Dict

from .base import FeatureExtractor
from src.models.processors import BaseProcessor

class MAE(FeatureExtractor):
    """
    Wrapper for ViT-MAE encoder to produce dense patch features
    """
    def __init__(self, backbone: str="base", preprocess_images: bool=True):
        """
        Args:
            backbone (str): Backbone architecture (e.g. "base"). Check models README
            preprocess_images (bool): Whether to preprocess images inside the forward pass. Default = True
        """
        super().__init__()
        possible_backbones = ("base", "large", "huge")
        assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)

        model_id = f"facebook/vit-mae-{backbone}"
        self.model = ViTMAEModel.from_pretrained(model_id)
        self.patch_size = self.model.config.patch_size
        self.feature_dim = self.model.config.hidden_size
        self.model.config.mask_ratio = 0.0  # disable masking
        self.preprocess_images = preprocess_images
        if self.preprocess_images:
            self.processor = BaseProcessor(patch_size=self.patch_size, normalize=True)

    def forward_features(self, images: torch.Tensor, unflatten_patches: bool=False) -> Dict[str, torch.Tensor]:
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
        # technically one should refrain from passing batches of shape B, S, C, H, W, but some
        # pipelines test models which take in both shape types 
        if len(images.shape) == 5:
            b, s, c, h, w = images.shape
            images = images.reshape(b * s, c, h, w)
            
        if self.preprocess_images:
            images = self.processor(images)

        _, _, h, w = images.shape
        num_patches_h, num_patches_w = h // self.patch_size, w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        outputs = self.model(pixel_values=images, interpolate_pos_encoding=True)
        last_hidden_states = outputs.last_hidden_state

        patch_embeds = last_hidden_states[:, -num_patches:, :]
        if unflatten_patches:
            patch_embeds = patch_embeds.unflatten(1, (num_patches_h, num_patches_w))
        
        output_dict = {
            "x_norm": last_hidden_states,
            "x_norm_clstoken": last_hidden_states[:, 0],
            "x_norm_patchtokens": patch_embeds
        }
        return output_dict
