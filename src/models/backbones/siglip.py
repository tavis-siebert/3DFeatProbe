import torch
import torch.nn as nn
from typing import Dict, Tuple
from transformers import Siglip2VisionModel, AutoProcessor

from src.utils.image_utils import center_pad

class SigLIPProcessor(nn.Module):
    def __init__(self, model_id: str, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.processor = AutoProcessor.from_pretrained(model_id)
    
    def forward(self, images: torch.Tensor) -> Tuple[Dict, int, int]:
        """
        Normalize images before feeding them into DINO models.
        
        Args:
            images (torch.Tensor): Batch of images of shape (B, 3, H, W).
                Can be in range [0, 255] or [0, 1].

        Returns:
            the BatchFeature dict from SigLIP's image processor
            tuple[int, int] number of patches in each spatial dimension
        """
        # scale to [0, 1] if necessary
        if images.max() > 1.0:
            images = images / 255.0
        
        # use HF processor to avoid shape mismatches
        # TODO: you should pad AFTER normalizing, so maybe use calculate_pad_hw and use the built-in resize of processor
        images = center_pad(images, self.patch_size)
        _, _, h, w = images.shape
        num_patches_h, num_patches_w = h // self.patch_size, w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        input = self.processor(
            images, 
            return_tensors="pt", 
            do_resize=False, 
            do_rescale=False,
            patch_size=self.patch_size,
            max_num_patches=num_patches
        )
        return input, num_patches_h, num_patches_w


class SigLIP2(nn.Module):
    """
    SigLIP2 (image encoder) model class
    """
    def __init__(self, backbone: str="base", preprocess_images: bool=True):
        """
        Args:
            backbone (str): Backbone architecture (e.g. "base"). Check models README
            preprocess_images (bool): Whether to preprocess images inside the forward pass. Default = True
        """
        super().__init__()

        possible_backbones = ("base", "so400m")
        assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)

        model_id = f"google/siglip2-{backbone}-patch16-naflex"
        self.model = Siglip2VisionModel.from_pretrained(model_id)
        self.patch_size = self.model.config.patch_size
        self.feature_dim = self.model.config.hidden_size
        self.processor = SigLIPProcessor(model_id=model_id, patch_size=self.patch_size)

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
        input, num_patches_h, num_patches_w = self.processor(images)
        num_patches = num_patches_h * num_patches_w

        outputs = self.model(**input)
        last_hidden_states = outputs.last_hidden_state
        
        output_dict = {
            "full_embeds": last_hidden_states,
            "cls_token": last_hidden_states[:, 0],
            "patch_embeds": last_hidden_states[:, -num_patches:, :].unflatten(1, (num_patches_h, num_patches_w))
        }
        return output_dict