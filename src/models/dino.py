import torch
import torch.nn as nn
import timm
from transformers import AutoModel

from src.models.processor import BaseProcessor

class DINOv2(nn.Module):
    """
    DINOv2 model class.
    """
    def __init__(self, backbone: str="base", with_registers=False, preprocess_images: bool=True):
        """
        Args:
            backbone (str): Backbone architecture (e.g. "base"). Check models README
            with_registers (bool): Whether to use DINOv2 with register tokens. Default = False.
            preprocess_images (bool): Whether to preprocess images inside the forward pass. Default = True
        """
        super().__init__()

        # uses timm
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
        #TODO: might need to interpolate pos embeds instead to handle dynamic input
        self.preprocess_images = preprocess_images
        if self.preprocess_images:
            self.processor = BaseProcessor(resize=True, resize_size=518, patch_size=self.patch_size, normalize=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W).

        Returns:
            feature map (torch.Tensor) of final hidden layer (B, H_patch, W_patch, D)
                We return the 2D spatial map because it's easier to go from unflattened to flattened than vise versa
        """
        if self.preprocess_images:
            images = self.processor(images)

        _, _, h, w = images.shape
        num_patches_h, num_patches_w = h // self.patch_size, w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        output = self.model.forward_features(images) # (B, 1 + num_register_tokens + num_patches, self.feature_dim)
        feature_map = output[:, -num_patches:, :].unflatten(1, (num_patches_h, num_patches_w))
        return feature_map
    

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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W).
        Returns:
            feature map (torch.Tensor) of final hidden layer (B, H_patch, W_patch, D)
                We return the 2D spatial map because it's easier to go from unflattened to flattened than vise versa
        """
        if self.preprocess_images:
            images = self.processor(images)

        _, _, h, w = images.shape
        num_patches_h, num_patches_w = h // self.patch_size, w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        output = self.model.forward_features(images) # (B, 1 + num_register_tokens + num_patches, self.feature_dim)
        feature_map = output[:, -num_patches:, :].unflatten(1, (num_patches_h, num_patches_w))
        return feature_map