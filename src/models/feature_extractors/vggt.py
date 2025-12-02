# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from typing import Dict, List
from vggt.models.aggregator import slice_expand_and_flatten

from src.models.vggt import VGGT
from .base import FeatureExtractor
from src.models.utils import load_checkpoint
from src.utils.image_utils import center_pad

class VGGTFeatureExtractor(FeatureExtractor):
    """
    Convenience class to extract all intermediate layers of VGGT
    NOTE: only extracts aggregator and not head-specific layers
    """
    def __init__(
        self,
        vggt_config: Dict,
        checkpoint_path: str=None,
        layer_types: List[str]=["all"],
    ):
        """
        Args (see VGGT class for full list):
            vggt_config (Dict): 
            layer_type (List[strr]): Specify one or more of the following
                - "patch_embed": the final layer of the patch_embed encoder model
                - "frame": all local attention layers within the aggregator
                - "global": all global attention layers within the aggregator
                - "all": all layers (patch_embed final, all local and global attention layers)
        """
        super().__init__()

        if checkpoint_path:
            if vggt_config:
                self.model = VGGT(**vggt_config)
            else:
                self.model = VGGT()

            load_checkpoint(self.model, checkpoint_path)
        else:
            self.model = VGGT.from_pretrained("facebook/VGGT-1B")

        possible_layer_types = ("patch_embed", "frame", "global", "all")
        assert all([layer in possible_layer_types for layer in layer_types]), \
            f"Unrecognized layer type found in {layer_types}. Possible layer types are {possible_layer_types}"

        self.layer_types = layer_types

    def forward_features(self, images: torch.Tensor, unflatten_patches=False) -> Dict:
        """
        Extract patch features from patch embedding model + all attention blocks
        
        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W]
            
        Returns:
            Dict mapping the layer name to the patch embeddings of shape [B, S, P, D] where D is flexible
                (depends on layer) (or shape [B, S, H_p, W_p, D] if unflattened is True)
        """
        output_dict = {}
        layers_contain = lambda l_name: l_name in self.layer_types or "all" in self.layer_types

        # Ensure batch dimension
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, S, C_in, H, W = images.shape

        # Extract patch embeddings
        images = images.view(B*S, C_in, H, W)
        if H % self.model.patch_size != 0 or W % self.model.patch_size != 0:
            images = center_pad(images, self.model.patch_size)
            _, _, H, W = images.shape # update H, W for dimension expansion later

        patch_tokens = self.model.aggregator.patch_embed(images)  # B*S, P, C
        
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
            
        H_p = H // self.model.patch_size
        W_p = W // self.model.patch_size

        if layers_contain("patch_embed"):
            if unflatten_patches: 
                output_dict["patch_embed"] = {
                    "x_norm_patchtokens": patch_tokens.unflatten(1, (H_p, W_p))
                }
            else: 
                output_dict["patch_embed"] = {
                    "x_norm_patchtokens": patch_tokens
                }
        
        # Expand camera and register tokens
        camera_token = slice_expand_and_flatten(self.model.aggregator.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.model.aggregator.register_token, B, S)
        
        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.model.aggregator.rope is not None:
            pos = self.model.aggregator.position_getter(B * S, H_p, W_p, device=images.device)
            
        if self.model.aggregator.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.model.aggregator.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
            
        # Update P after adding special tokens
        _, P, C = tokens.shape
        
        # Process through attention blocks and collect features
        frame_idx = 0
        global_idx = 0
        
        for layer_num in range(self.model.aggregator.aa_block_num):
            for attn_type in self.model.aggregator.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self.model.aggregator._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self.model.aggregator._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # Concatenate frame and global intermediates
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                
                # Extract only patch tokens (exclude camera and register tokens)
                patch_features = concat_inter[:, :, self.model.aggregator.patch_start_idx:, :]  # [B, S, P_patch, 2*C]

                # Handle user specified layer types
                C_out = patch_features.shape[-1]

                if layers_contain("frame"):
                    if unflatten_patches: 
                        patch_features = patch_features.view(B, S, H_p, W_p, -1)
                    output_dict[f"aggregator_frame-{layer_num}-{i}"] = {
                        "x_norm_patchtokens": patch_features[..., :C_out // 2]
                    }

                if layers_contain("global"):
                    if unflatten_patches: 
                        patch_features = patch_features.view(B, S, H_p, W_p, -1)
                    output_dict[f"aggregator_global-{layer_num}-{i}"] = {
                        "x_norm_patchtokens": patch_features[..., -(C_out // 2):]
                    }

        return output_dict