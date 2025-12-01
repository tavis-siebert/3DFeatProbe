import torch
import torch.nn as nn 
from abc import ABC, abstractmethod
from typing import Dict, Union

class FeatureExtractor(nn.Module, ABC):
    """
    Base class ensuring all feature_extractor models output a standardized 
    schema in their forward() pass.

    The schema supports
        - output_dict (Dict[str, torch.Tensor]): convenient views of last hidden states for downstream use
            ```
            {
              "x_norm": # the last hidden state in full (including CLS and possible register or other tokens)
              "x_norm_clstoken": # the CLS token for classification or semantic tasks
              "x_norm_patchtokens": # the patch embeddings
            }
            ```
        - output_dict (Dict[Dict[str, torch.Tensor]]): layer-wise feature dicts
            ```
            {
              <layer_i>: {
                "x_norm_patchtokens": # the patch embeddings
                ... # more keys like "x_norm_clstoken" are permissible, but not expected to save memory
              },
              ...
            }
            ```
    """

    @abstractmethod
    def forward_features(
        self, images: torch.Tensor, 
        unflatten_patches: bool=False, 
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
        """
        **Subclasses must implement this**
        
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W)
            unflatten_patches (bool): Whether the patch embeddings are returned as a contiguous sequence 
                                    of patches or a 2D spatial map. Default is flattened (False).

        Returns:
            One of the following

            output_dict (Dict[str, torch.Tensor]): convenient views of last hidden states for downstream use
                ```
                {
                    "x_norm": # the last hidden state in full (including CLS and possible register or other tokens)
                    "x_norm_clstoken": # the CLS token for classification or semantic tasks
                    "x_norm_patchtokens": # the patch embeddings
                }
                ```
            output_dict (Dict[Dict[str, torch.Tensor]]): layer-wise feature dicts
                ```
                {
                    <layer_i>: {
                        "x_norm_patchtokens": # the patch embeddings
                        ... # more keys like "x_norm_clstoken" are permissible, but not expected to save memory
                    },
                ...
                }
                ```
        """
        pass

    def validate_output_schema(self, out_dict) -> str:
        """
        Validates the output schema and returns the schema type for convenience

        Returns:
            out (str): the schema type as a string either "single" or "multi"
        """
        # Single-layer schema
        required_single = {
            "x_norm",
            "x_norm_clstoken",
            "x_norm_patchtokens",
        }

        if isinstance(out_dict, dict) and required_single.issubset(out_dict.keys()):
            for key in required_single:
                if not isinstance(out_dict[key], torch.Tensor):
                    raise TypeError(f"Expected '{key}' to be a torch.Tensor.")
            return "single"

        # Multi-layer schema
        if isinstance(out_dict, dict):
            if all(isinstance(v, dict) for v in out_dict.values()):
                for layer_name, layer_dict in out_dict.items():
                    if "x_norm_patchtokens" not in layer_dict:
                        raise KeyError(
                            f"Layer '{layer_name}' missing required key 'x_norm_patchtokens'."
                        )

                    # Validate tensor-ness
                    for key, value in layer_dict.items():
                        if not isinstance(value, torch.Tensor):
                            raise TypeError(
                                f"In layer '{layer_name}', key '{key}' must be a torch.Tensor."
                            )
                return  "multi"

        # Neither schema matched
        raise ValueError(
            "Output schema invalid. Must be either:\n"
            "1. Single-layer dict with keys: "
            "{'x_norm', 'x_norm_clstoken', 'x_norm_patchtokens'}\n"
            "2. Multi-layer dict mapping layer → dict containing at least 'x_norm_patchtokens'."
        )

    def forward(
        self, 
        images: torch.Tensor, 
        unflatten_patches: bool=False, 
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
        """
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W) or (B, S, C, H, W) 
            unflatten_patches (bool): Whether the patch embeddings are returned as a contiguous sequence 
                                    of patches or a 2D spatial map. Default is flattened (False).

        Returns:
            One of the following

            output_dict (Dict[str, torch.Tensor]): convenient views of last hidden states for downstream use
                ```
                {
                    "x_norm": # the last hidden state in full (including CLS and possible register or other tokens)
                    "x_norm_clstoken": # the CLS token for classification or semantic tasks
                    "x_norm_patchtokens": # the patch embeddings
                }
                ```
            output_dict (Dict[str, Dict[str, torch.Tensor]]): layer-wise feature dicts
                ```
                {
                    <layer_i>: {
                        "x_norm_patchtokens": # the patch embeddings
                        ... # more keys like "x_norm_clstoken" are permissible, but not expected to save memory
                    },
                ...
                }
                ```
        """
        out_dict = self.forward_features(images, unflatten_patches, **kwargs)
        self.validate_output_schema(out_dict)
        return out_dict