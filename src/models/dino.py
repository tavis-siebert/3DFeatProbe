
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

class BaseDINO(nn.Module):
    def __init__(
        self, 
        dino_name: str="dinov2",
        backbone: str="vitb14"
    ):
        """
        Base class for DINO models.

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
            possible_backbones = [
                "vits14", 
                "vitb14", 
                "vitl14",
                "vitg14", 
                "vits14_reg",
                "vitb14_reg",
                "vitl14_reg",
                "vitg14_reg",
            ]
            assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)

            feature_dims = {
                "vits14": 384,
                "vitb14": 768,
                "vitl14": 1024,
                "vitg14": 1536
            }

            self.model = torch.hub.load('facebookresearch/dinov2', backbone)
            self.patch_size = 14
            self.feature_dim = feature_dims[backbone]
        elif dino_name == "dinov3":
            possible_backbones = [
                "vits16",
                "vits16plus",
                "vitb16",
                "vitl16",
                "vith16plus",
                "vit7b16",
                # TODO: support ConvNext (e.g. handle patch size, feature dim, etc)
                # "convnext-tiny",
                # "convnext-small",
                # "convnext-base",
                # "convnext-large"
            ]
            assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)

            self.model = AutoModel.from_pretrained(f"facebook/dinov3-{backbone}-pretrain-lvd1689m")
            self.patch_size = self.model.config.patch_size # but should be 16 for all ViTs
            self.feature_dim = self.model.config.hidden_size
        else:
            raise Exception("DINO name must be one of ['dino', 'dinov2', 'dinov3']")

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Base forward method.

        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Additional arguments for specific model behavior.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class DINOv2(BaseDINO):
    def __init__(
        self, 
        backbone: str="vitb14"
    ):
        """
        DINOv2 model class.

        Args:
            backbone (str): Backbone architecture (e.g. "vitb14"). Check model card at https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
        """
        super().__init__(dino_name="dinov2", backbone=backbone)
        # TODO: any dinov2 or torchhub specific stuff

    def forward(self, x: torch.Tensor, **kwargs):
        # TODO: extract features (there might be register vs non-register differences)
        raise NotImplementedError("Forward not implemented yet")

class DINOv3(BaseDINO):
    def __init__(
        self, 
        backbone: str="vitb16"
    ):
        """
        DINOv3 model class.

        Args:
            backbone (str): Backbone architecture (e.g. "vitb16"). Check model card at https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md
        """
        super().__init__(dino_name="dinov3", backbone=backbone)
        # TODO: any dinov3 or HF specific stuff
    
    def forward(self, x: torch.Tensor, **kwargs):
        # TODO: extract features (there might be register vs non-register differences)
        raise NotImplementedError("Forward not implemented yet")