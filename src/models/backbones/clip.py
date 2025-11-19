import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from typing import Dict

from src.models.processors import BaseProcessor

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class CLIP(nn.Module):
    """
    CLIP (image encoder) model class
    """
    def __init__(self, backbone: str="vit-base-patch16", preprocess_images: bool=True):
        """
        Args:
            backbone (str): Backbone architecture. See model README for details
            preprocess_images (bool): Whether to preprocess images inside the forward pass. Default = True
        """
        super().__init__()

        possible_backbones = ("vit-large-patch14", "vit-base-patch16", "vit-base-patch32")
        assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)
        
        self.model = CLIPVisionModel.from_pretrained(f"openai/clip-{backbone}")
        self.patch_size = self.model.config.patch_size
        self.feature_dim = self.model.config.hidden_size
        self.preprocess_images = preprocess_images
        if self.preprocess_images:
            self.processor = BaseProcessor(patch_size=self.patch_size, normalize=True, mean=CLIP_MEAN, std=CLIP_STD)
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W).

        Returns:
            output_dict (Dict[str, torch.Tensor]): convenient views of last hidden states for downstream use
                "full_embeds": the last hidden state in full (including CLS and possible register or other tokens)
                "cls_token": the CLS token for classification or semantic tasks
                "patch_embeds": the patch embeddings as a 2D spatial map.
                                We return the 2D view as it's easier to flatten later than unflatten without knowledge of num_patches in H, W
        """
        if self.preprocess_images:
            images = self.processor(images)
        
        _, _, h, w = images.shape
        num_patches_h, num_patches_w = h // self.patch_size, w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        outputs = self.model(pixel_values=images, interpolate_pos_encoding=True)
        last_hidden_states = outputs.last_hidden_state
        
        output_dict = {
            "full_embeds": last_hidden_states,
            "cls_token": last_hidden_states[:, 0],
            "patch_embeds": last_hidden_states[:, -num_patches:, :].unflatten(1, (num_patches_h, num_patches_w))
        }
        return output_dict
