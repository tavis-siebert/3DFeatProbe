
import torch
import torch.nn as nn
from transformers import AutoModel

from src.utils.image_utils import center_pad

class BaseDINO(nn.Module):
    """
    Base class for DINO models.
    """
    def __init__(
        self, 
        dino_name: str="dinov2",
        backbone: str="base",
        with_registers: bool=False,
    ):
        """
        Args:
            dino_name (str): Name of the DINO model. Options: 'dino', 'dinov2', 'dinov3'.
            backbone (str): Backbone architecture. Options depend on dino_name. Check model cards on respective repos for details.
        """
        super().__init__()

        self.model = None
        self.patch_size = None
        self.feature_dim = None

        if dino_name == "dino":
            # TODO: implement
            raise NotImplementedError("Yet to implement original DINO")
        elif dino_name == "dinov2":
            '''
            # torchhub Version
            # Doesn't work due to ssl or proxy or something on euler
            possible_backbones = [
                "vits14", 
                "vitb14", 
                "vitl14",
                "vitg14", 
            ]
            assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)

            feature_dims = {
                "vits14": 384,
                "vitb14": 768,
                "vitl14": 1024,
                "vitg14": 1536
            }

            if with_registers: backbone += "_reg" 

            self.model = torch.hub.load('facebookresearch/dinov2', f"dinov2_{backbone}")
            self.patch_size = 14
            self.feature_dim = feature_dims[backbone]
            '''
            # huggingface version
            possible_backbones = [
                "small", 
                "base", 
                "large",
                "giant", 
            ]
            assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)

            if with_registers: backbone = f"with-registers-{backbone}"
            full_model_path = f"facebook/dinov2-{backbone}"

            self.model = AutoModel.from_pretrained(full_model_path)
            self.patch_size = self.model.config.patch_size
            self.feature_dim = self.model.config.hidden_size
        elif dino_name == "dinov3":
            possible_backbones = [
                "vits16",
                "vits16plus",
                "vitb16",
                "vitl16",
                "vith16plus",
                "vit7b16"
                # TODO: support ConvNext (e.g. handle patch size, feature dim, etc)
                # "convnext-tiny",
                # "convnext-small",
                # "convnext-base",
                # "convnext-large"
            ]
            assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)

            full_model_path = f"facebook/dinov3-{backbone}-pretrain-lvd1689m"

            self.model = AutoModel.from_pretrained(full_model_path)
            self.patch_size = self.model.config.patch_size # but should be 16 for all ViTs
            self.feature_dim = self.model.config.hidden_size
        else:
            raise Exception("DINO name must be one of ['dino', 'dinov2', 'dinov3']")

    def forward(self, images: torch.Tensor, **kwargs):
        """
        Template forward method.

        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W).
            **kwargs: Additional arguments for specific model behavior.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class DINOv2(BaseDINO):
    """
    DINOv2 model class.
    """
    def __init__(self, backbone: str="base", with_registers=False):
        """
        Args:
            backbone (str): Backbone architecture (e.g. "base"). Check model card at https://huggingface.co/collections/facebook/dinov2-6526c98554b3d2576e071ce3
            with_registers (bool): Whether to use DINOv2 with register tokens. Default = False.
        """
        super().__init__(dino_name="dinov2", backbone=backbone, with_registers=with_registers)
    
    def _apply_transforms(self, images: torch.Tensor) -> torch.Tensor:
        """
        Normalize images for DINOv2 before feeding them into the model.
        
        Args:
            images (torch.Tensor): Batch of images of shape (B, 3, H, W).
                Can be in range [0, 255] or [0, 1].

        Returns:
            torch.Tensor: Normalized images of shape (B, 3, H, W).
        """
        # scale to [0, 1] if necessary
        if images.max() > 1.0:
            images = images / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)

        return (images - mean) / std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W).
            **kwargs: Additional arguments for specific model behavior.
        Returns:
            feature map (torch.Tensor) of final hidden layer (B, H_patch, W_patch, D)
                We return the 2D spatial map because it's easier to go from unflattened to flattened than vise versa
        """
        # pad until divisible by patch size
        images = center_pad(images, self.patch_size)
        images = self._apply_transforms(images)

        _, _, h, w = images.shape
        num_patches_h, num_patches_w = h // self.patch_size, w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        outputs = self.model(pixel_values=images)
        last_hidden_states = outputs[0] # (B, 1 + num_register_tokens + num_patches, self.feature_dim)
        feature_map = last_hidden_states[:, -num_patches:, :].unflatten(1, (num_patches_h, num_patches_w))

        return feature_map

class DINOv3(BaseDINO):
    """
    DINOv3 model class.
    """
    def __init__(self, backbone: str="vitb16"):
        """
        Args:
            backbone (str): Backbone architecture (e.g. "vitb16"). Check model card at https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md
        """
        super().__init__(dino_name="dinov3", backbone=backbone)

    def forward(self, images: torch.Tensor, **kwargs):
        # TODO: extract features (there might be register vs non-register differences)
        raise NotImplementedError("Forward not implemented yet")