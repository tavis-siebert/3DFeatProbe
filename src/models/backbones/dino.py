import torch
import torch.nn as nn
import timm
from typing import Dict
from transformers import AutoModel

from src.models.processors import BaseProcessor

class DINOv2(nn.Module):
    """
    DINOv2 model class.
    """
    def __init__(
            self, 
            backbone: str="base", 
            use_timm: bool=False,
            with_registers=False, 
            preprocess_images: bool=True):
        """
        Args:
            backbone (str): Backbone architecture (e.g. "base"). Check models README
            use_timm (bool): If true, uses timm model. Default = False
            with_registers (bool): Whether to use DINOv2 with register tokens. Default = False
            preprocess_images (bool): Whether to preprocess images inside the forward pass. Default = True
        """
        super().__init__()

        # choose implementation (slight differences with state_dict, but same weights)
        self.use_timm = use_timm
        if self.use_timm:
            self._init_with_timm(backbone, with_registers)
        else:
            self._init_with_hf(backbone, with_registers)
        
        #TODO: might need to interpolate pos embeds instead to handle dynamic input
        self.preprocess_images = preprocess_images
        if self.preprocess_images:
            self.processor = BaseProcessor(resize=True, resize_size=518, patch_size=self.patch_size, normalize=True)
    
    def _init_with_timm(self, backbone, with_registers):
        """Initialize with timm"""
        possible_backbones = (
            "small", 
            "base", 
            "large",
            "giant", 
        )
        assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)

        full_model_path = f"vit_{backbone}_patch14_reg4_dinov2.lvd142m" if with_registers else f"vit_{backbone}_patch14_dinov2.lvd142m"

        self.model = timm.create_model(full_model_path, pretrained=True)
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.feature_dim = self.model.embed_dim
    
    def _init_with_hf(self, backbone, with_registers):
        "Initialize with official facebook version on huggingface"
        possible_backbones = (
            "small", 
            "base", 
            "large",
            "giant", 
        )
        assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)

        if with_registers: backbone = f"with-registers-{backbone}"
        full_model_path = f"facebook/dinov2-{backbone}"

        self.model = AutoModel.from_pretrained(full_model_path)
        self.patch_size = self.model.config.patch_size
        self.feature_dim = self.model.config.hidden_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
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

        if self.use_timm:
            last_hidden_states = self.model.forward_features(images) # (B, 1 + num_register_tokens + num_patches, self.feature_dim)
        else:
            outputs = self.model(pixel_values=images)
            last_hidden_states = outputs[0]
        
        output_dict = {
            "full_embeds": last_hidden_states,
            "cls_token": last_hidden_states[:, 0],
            "patch_embeds": last_hidden_states[:, -num_patches:, :].unflatten(1, (num_patches_h, num_patches_w))
        }
        return output_dict
    

class DINOv3(nn.Module):
    """
    DINOv3 model class.
    """
    def __init__(self, backbone: str="base", preprocess_images: bool=True):
        """
        Args:
            backbone (str): Backbone architecture (e.g. "base"). Check models README
            preprocess_images (bool): Whether to preprocess images inside the forward pass. Default = True
        """
        super().__init__()

        # uses timm
        possible_backbones = (
            "small",
            "small_plus",
            "base",
            "large",
            "huge",
            "huge_plus",
            "7b"
            # TODO: support ConvNext (e.g. handle patch size, feature dim, etc)
            # "convnext-tiny",
            # "convnext-small",
            # "convnext-base",
            # "convnext-large"
        )
        assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)

        full_model_path = f"vit_{backbone}_patch16_dinov3.lvd1689m"

        self.model = timm.create_model(full_model_path, pretrained=True)
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.feature_dim = self.model.embed_dim
        self.preprocess_images = preprocess_images
        if self.preprocess_images:
            self.processor = BaseProcessor(patch_size=self.patch_size, normalize=True)

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

        last_hidden_states = self.model.forward_features(images) # (B, 1 + num_register_tokens + num_patches, self.feature_dim)
        
        output_dict = {
            "full_embeds": last_hidden_states,
            "cls_token": last_hidden_states[:, 0],
            "patch_embeds": last_hidden_states[:, -num_patches:, :].unflatten(1, (num_patches_h, num_patches_w))
        }
        return output_dict