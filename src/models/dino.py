import torch
import torch.nn as nn
import timm
from transformers import AutoModel

from src.utils.image_utils import center_pad

class DINOProcessor(nn.Module):
    #TODO: Still not sure whether this is better or trying to hack around the processor on hf
    # I am breaking DRY pretty badly as all my ViT mdodels seem to do this
    # The big thing that determines whether I make a base ViTProcessor class is whether I do this center pad on every image
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Normalize images before feeding them into DINO models.
        
        Args:
            images (torch.Tensor): Batch of images of shape (B, 3, H, W).
                Can be in range [0, 255] or [0, 1].

        Returns:
            torch.Tensor: Normalized images of shape (B, 3, H, W).
        """
        # scale to [0, 1] if necessary
        if images.max() > 1.0:
            images = images / 255.0
        
        # center pad to ensure divisibility w/ patch size
        images = center_pad(images, self.patch_size)

        # normalie with ImageNet mean, std
        return (images - self.mean) / self.std

class DINOv2(nn.Module):
    """
    DINOv2 model class.
    """
    def __init__(self, backbone: str="base", with_registers=False):
        """
        Args:
            backbone (str): Backbone architecture (e.g. "base"). Check models README
            with_registers (bool): Whether to use DINOv2 with register tokens. Default = False.
        """
        super().__init__()
        
        # uses hf transformers
        possible_backbones = (
            "small", 
            "base", 
            "large",
            "giant", 
        )
        assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)

        if with_registers: backbone = f"with-registers-{backbone}"
        full_model_path = f"facebook/dinov2-{backbone}"

        self.model = AutoModel.from_pretrained(full_model_path) #TODO: set eval() if we don't FT
        self.patch_size = self.model.config.patch_size
        self.feature_dim = self.model.config.hidden_size
        self.processor = DINOProcessor(self.patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W).

        Returns:
            feature map (torch.Tensor) of final hidden layer (B, H_patch, W_patch, D)
                We return the 2D spatial map because it's easier to go from unflattened to flattened than vise versa
        """
        images = self.processor(images)

        _, _, h, w = images.shape
        num_patches_h, num_patches_w = h // self.patch_size, w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        outputs = self.model(pixel_values=images)
        last_hidden_states = outputs[0] # (B, 1 + num_register_tokens + num_patches, self.feature_dim)
        feature_map = last_hidden_states[:, -num_patches:, :].unflatten(1, (num_patches_h, num_patches_w))
        return feature_map

class DINOv3(nn.Module):
    """
    DINOv3 model class.
    """
    def __init__(self, backbone: str="base"):
        """
        Args:
            backbone (str): Backbone architecture (e.g. "base"). Check models README
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
        self.processor = DINOProcessor(self.patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W).

        Returns:
            feature map (torch.Tensor) of final hidden layer (B, H_patch, W_patch, D)
                We return the 2D spatial map because it's easier to go from unflattened to flattened than vise versa
        """
        images = self.processor(images)

        _, _, h, w = images.shape
        num_patches_h, num_patches_w = h // self.patch_size, w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        output = self.model.forward_features(images) # (B, 1 + num_register_tokens + num_patches, self.feature_dim)
        feature_map = output[:, -num_patches:, :].unflatten(1, (num_patches_h, num_patches_w))
        return feature_map