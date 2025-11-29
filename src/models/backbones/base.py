import torch
import torch.nn as nn 
from abc import ABC, abstractmethod
from typing import Dict

class FeatureBackbone(nn.Module, ABC):
    """
    Base class ensuring all backbone models output a standardized 
    schema in their forward() pass.

    The schema is 
        output_dict (Dict[str, torch.Tensor]):
            - "x_norm": the last hidden state in full (including CLS and possible register or other tokens)
            - "x_norm_clstoken": the CLS token for classification or semantic tasks
            - "x_norm_patchtokens": the patch embeddings
    """
    required_keys = ("x_norm", "x_norm_clstoken", "x_norm_patchtokens")

    @abstractmethod
    def forward_features(self, images: torch.Tensor, unflatten_patches: bool=False, **kwargs) -> Dict[str, torch.Tensor]:
        """
        **Subclasses must implement this**
        
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W)
            unflatten_patches (bool): Whether the patch embeddings are returned as a contiguous sequence 
                                    of patches or a 2D spatial map. Default is flattened (False).

        Returns:
            output_dict (Dict[str, torch.Tensor]): convenient views of last hidden states for downstream use
                "x_norm": the last hidden state in full (including CLS and possible register or other tokens)
                "x_norm_clstoken": the CLS token for classification or semantic tasks
                "x_norm_patchtokens": the patch embeddings
        """
        pass

    def forward(self, images: torch.Tensor, unflatten_patches: bool=False, **kwargs):
        """
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W)
            unflatten_patches (bool): Whether the patch embeddings are returned as a contiguous sequence 
                                    of patches or a 2D spatial map. Default is flattened (False).

        Returns:
            output_dict (Dict[str, torch.Tensor]): convenient views of last hidden states for downstream use
                "x_norm": the last hidden state in full (including CLS and possible register or other tokens)
                "x_norm_clstoken": the CLS token for classification or semantic tasks
                "x_norm_patchtokens": the patch embeddings
        """

        out_dict = self.forward_features(images, unflatten_patches, **kwargs)

        # Validate schema
        for key in self.required_keys:
            if key not in out_dict:
                raise KeyError(f"Missing required key '{key}' in forward output.")
        
        return out_dict