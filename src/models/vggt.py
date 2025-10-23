# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Dict
from vggt.models.vggt import VGGT
from vggt.models.aggregator import slice_expand_and_flatten

from src.utils.image_utils import center_pad

class VGGTFeatureExtractor(nn.Module):
    def __init__(self, backbone: str="VGGT-1B", layer_type: str="all"):
        """
        Args:
            backbone (str): Backbone architecture (currently only "VGGT-1B" model)
            layer_type (str): Whether to return only frame, only global, or concatenated (all) embeddings
                              Options [`"frame"`, `"global"`, `"all"`]. Defaults to `"all"`
        """
        super().__init__()

        possible_backbones = ("VGGT-1B")
        possible_layer_types = ("frame", "global", "all")
        assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)
        assert layer_type in possible_layer_types, "Layer type must be one of {}".format(possible_layer_types)

        self.vggt = VGGT.from_pretrained(f"facebook/{backbone}")
        self.aggregator = self.vggt.aggregator
        self.patch_size = self.aggregator.patch_size
        self.layer_type = layer_type

    # TODO: I also basically just copied Aggregator. It might be cleaner to use a processor to downsize to H, W divisible 
    # and then just do a self.aggregator pass and use the start idx, resize size, and patch size to find the patch tokens
    def forward(self, images: torch.Tensor) -> Dict:
        """
        Extract patch features from patch embedding model + all attention blocks
        
        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W]
            
        Returns:
            Dict mapping the layer name to the patch embeddings of shape [B, S, H_p, W_p, D] where D is flexible
                (depends on layer)
        """
        output_dict = {}

        # Ensure batch dimension
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        B, S, C_in, H, W = images.shape

        # Normalize images
        images = (images - self.aggregator._resnet_mean) / self.aggregator._resnet_std
        
        # Extract patch embeddings
        # TODO: can do this in all in processor, but then need to get H, W after resize
        images = images.view(B*S, C_in, H, W)
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            images = center_pad(images, self.patch_size)
            _, _, H, W = images.shape # update H, W for dimension expansion later
        
        patch_tokens = self.aggregator.patch_embed(images)
        
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
            
        _, P, C = patch_tokens.shape
        H_p = H // self.patch_size
        W_p = W // self.patch_size
        output_dict["patch_tokens"] = patch_tokens.view(B, S, H_p, W_p, C)
        
        # Expand camera and register tokens
        camera_token = slice_expand_and_flatten(self.aggregator.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.aggregator.register_token, B, S)
        
        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.aggregator.rope is not None:
            pos = self.aggregator.position_getter(B * S, H_p, W_p, device=images.device)
            
        if self.aggregator.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.aggregator.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
            
        # Update P after adding special tokens
        _, P, C = tokens.shape
        
        # Process through attention blocks and collect features
        frame_idx = 0
        global_idx = 0
        
        for layer_num in range(self.aggregator.aa_block_num):
            for attn_type in self.aggregator.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self.aggregator._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self.aggregator._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # Concatenate frame and global intermediates
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                
                # Extract only patch tokens (exclude camera and register tokens)
                patch_features = concat_inter[:, :, self.aggregator.patch_start_idx:, :]  # [B, S, P_patch, 2*C]
                patch_features = patch_features.view(B, S, H_p, W_p, -1)

                # handle if user specifies specific layer type
                C_out = patch_features.shape[-1]
                if self.layer_type == "frame":
                    patch_features = patch_features[..., :C_out // 2]
                elif self.layer_type == "global":
                    patch_features = patch_features[..., -(C_out // 2):]
                
                output_dict[f"layer_{layer_num}-{i}"] = patch_features

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_dict