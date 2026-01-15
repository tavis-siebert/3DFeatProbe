import torch
import torch.nn as nn
import timm
import math
from typing import Dict
from transformers import AutoModel

from .base import FeatureExtractor
from src.models.processors import BaseProcessor
from src.models.utils import load_checkpoint

class DINOv2(FeatureExtractor):
    """
    DINOv2 model class.
    """
    def __init__(
        self, 
        checkpoint_path: str=None,
        backbone: str="base", 
        use_timm: bool=False,
        with_registers=False, 
        preprocess_images: bool=True
    ):
        """
        Args:
            checkpoint_path (str): the path to a specific checkpoint. By default all models
                                    are loaded from huggingface or torchhub pretrained weights unless
                                    a checkpoint is provided
            backbone (str): Backbone architecture (e.g. "base"). Check models README
            use_timm (bool): If true, uses timm model. Default = False
            with_registers (bool): Whether to use DINOv2 with register tokens. Default = False
            preprocess_images (bool): Whether to preprocess images inside the forward pass. Default = True
        """
        super().__init__()

        self.use_timm = use_timm
        self.with_registers = with_registers
        if self.use_timm:
            self._init_with_timm(checkpoint_path, backbone, self.with_registers)
        else:
            self._init_with_hf(checkpoint_path, backbone, self.with_registers)
        
        self.preprocess_images = preprocess_images
        if self.preprocess_images:
            self.processor = BaseProcessor(patch_size=self.patch_size, normalize=True)
    
    def _init_with_timm(self, checkpoint_path, backbone, with_registers):
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
        if checkpoint_path:
            load_checkpoint(self.model, checkpoint_path)
        self.model.patch_embed.strict_img_size = False
        self.model.forward_features = lambda x, masks=None: self._timm_forward_features(x, masks=masks)

        self.patch_size = self.model.patch_embed.patch_size[0]
        self.embed_dim = self.model.embed_dim
        img_size = self.model.patch_embed.img_size
        if isinstance(img_size, tuple):
            if img_size[0] != img_size[1]:
                img_size = min(img_size)
            else:
                img_size = img_size[0]
        self.img_size = img_size
    
    def _init_with_hf(self, checkpoint_path, backbone, with_registers):
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
        if checkpoint_path:
            load_checkpoint(self.model, checkpoint_path)

        self.patch_size = self.model.config.patch_size
        self.embed_dim = self.model.config.hidden_size
        self.img_size = self.model.config.image_size

    def _interpolate_pos_embed(self, H, W, antialias=False, offset=0.1):
        """
        Robust interpolation that supports two pos_embed layouts:
         - pos_embed with class token included: shape (1, 1 + N, C)
         - pos_embed with only patch tokens:     shape (1, N, C)

        Returns a tensor shaped (1, 1 + N_new, C) if pos_embed originally included class,
        or (1, N_new, C) if not.
        """
        pos_embed = self.model.pos_embed

        # num patches target
        H_p = H // self.patch_size
        W_p = W // self.patch_size

        # pos_embed: (1, P, C)
        P = pos_embed.shape[1]
        C = pos_embed.shape[2]

        # decide whether pos_embed includes class token
        # NOTE: don't account for possibly more than CLS token, but I've never seen this case
        has_cls = False
        if P == 1 + int(math.sqrt(P - 1))**2:
            has_cls = True
        if has_cls:
            cls_pos = pos_embed[:, :1]
            patch_pos = pos_embed[:, 1:]
        else:
            cls_pos = None
            patch_pos = pos_embed 

        N = patch_pos.shape[1]
        M = int(math.sqrt(N))
        assert M * M == N, f"pos_embed grid not square: N={N}, M^2={M*M}"

        # reshape to (1, C, M, M)
        patch_pos = patch_pos.reshape(1, M, M, C).permute(0, 3, 1, 2)

        kwargs = {}
        if offset is not None:
            sx = float(W_p + offset) / M
            sy = float(H_p + offset) / M
            kwargs["scale_factor"] = (sy, sx)
        else:
            kwargs["size"] = (H_p, W_p)

        # call interpolate (bicubic like HF implementation)
        patch_pos = nn.functional.interpolate(
            patch_pos, 
            mode="bicubic", 
            antialias=antialias,
            **kwargs
        )

        # back to (1, H_p * W_p, C)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, C)

        if has_cls:
            return torch.cat((cls_pos, patch_pos), dim=1).to(pos_embed.dtype)
        else:
            return patch_pos.to(pos_embed.dtype)

    def _prepare_tokens(self, x, masks):
        """
        Prepare tokens with positional encoding resizing and cls token
        """
        B, C, H, W = x.shape

        x_patches = self.model.patch_embed(x)

        if masks is not None:
            x_patches = torch.where(masks.unsqueeze(-1), self.mask_token.to(x_patches.dtype).unsqueeze(0), x_patches)

        # Pos embed interp
        cls_token = self.model.cls_token.expand(B, -1, -1)  # (B,1,C)
        x_tokens = torch.cat((cls_token, x_patches), dim=1)  # (B, 1 + N, C)

        pos_embed_interp = self._interpolate_pos_embed(H, W)

        # handle case where model did not train pos_embed for cls token
        if pos_embed_interp.shape[1] == x_tokens.shape[1] - 1:
            cls_pos = torch.zeros(1, 1, pos_embed_interp.shape[2], dtype=pos_embed_interp.dtype, device=pos_embed_interp.device)
            pos_embed_interp = torch.cat((cls_pos, pos_embed_interp), dim=1)
        
        x_tokens = x_tokens + pos_embed_interp

        # Handle register tokens
        if self.with_registers:
            x_tokens = torch.cat((x_tokens[:, :1], self.model.reg_token.expand(B, -1, -1).to(x_tokens.dtype).to(x_tokens.device), x_tokens[:, 1:]), dim=1)

        return x_tokens

    def _timm_forward_features(self, x, masks):
        """
        Run forward across timm layers with proper positional encoding resize
        """
        x_tokens = self._prepare_tokens(x, masks)

        for blk in self.model.blocks:
            x_tokens = blk(x_tokens)

        x_norm = self.model.norm(x_tokens)
        return {
            "x_norm": x_norm,
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : 1 + self.model.num_reg_tokens],
            "x_norm_patchtokens": x_norm[:, 1 + self.model.num_reg_tokens:],
            "masks": masks,
        }

    def forward_features(self, images: torch.Tensor, unflatten_patches: bool=False) -> Dict[str, torch.Tensor]:
        """
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W).
            unflatten_patches (bool): Whether the patch embeddings are returned as a contiguous sequence 
                                    of patches or a 2D spatial map. Default is flattened (False).
        
        Returns:
            output_dict (Dict[str, torch.Tensor]): convenient views of last hidden states for downstream use
                "x_norm": the last hidden state in full (including CLS and possible register or other tokens)
                "x_norm_clstoken": the CLS token for classification or semantic tasks
                "x_norm_patchtokens": the patch embeddings
        """
        # technically one should refrain from passing batches of shape B, S, C, H, W, but some
        # pipelines test models which take in both shape types 
        if len(images.shape) == 5:
            b, s, c, h, w = images.shape
            images = images.reshape(b * s, c, h, w)

        if self.preprocess_images:
            images = self.processor(images)

        _, _, h, w = images.shape
        num_patches_h, num_patches_w = h // self.patch_size, w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        if self.use_timm:
            outputs = self.model.forward_features(images)
            x_norm = outputs["x_norm"]
            x_norm_clstoken = outputs["x_norm_clstoken"]
            x_norm_patchtokens = outputs["x_norm_patchtokens"]
        else:
            outputs = self.model(pixel_values=images)
            x_norm = outputs[0]
            x_norm_clstoken = x_norm[:, 0]
            x_norm_patchtokens = x_norm[:, -num_patches:, :]

        if unflatten_patches:
            x_norm_patchtokens = x_norm_patchtokens.unflatten(1, (num_patches_h, num_patches_w))
        
        # stick with VisionTransformer convention
        output_dict = {
            "x_norm": x_norm,
            "x_norm_clstoken": x_norm_clstoken,
            "x_norm_patchtokens": x_norm_patchtokens
        }
        return output_dict
    

class DINOv3(FeatureExtractor):
    """
    DINOv3 model class.
    """
    def __init__(self, checkpoint_path: str=None, backbone: str="base", preprocess_images: bool=True):
        """
        Args:
            checkpoint_path (str): the path to a specific checkpoint. By default all models
                                    are loaded from huggingface or torchhub pretrained weights unless
                                    a checkpoint is provided
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
        )
        assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)

        full_model_path = f"vit_{backbone}_patch16_dinov3.lvd1689m"

        self.model = timm.create_model(full_model_path, pretrained=True)
        if checkpoint_path:
            load_checkpoint(self.model, checkpoint_path)

        self.patch_size = self.model.patch_embed.patch_size[0]
        self.embed_dim = self.model.embed_dim
        img_size = self.model.patch_embed.img_size
        if isinstance(img_size, tuple):
            if img_size[0] != img_size[1]:
                img_size = min(img_size)
            else:
                img_size = img_size[0]
        self.img_size = img_size

        self.preprocess_images = preprocess_images
        if self.preprocess_images:
            self.processor = BaseProcessor(patch_size=self.patch_size, normalize=True)

    def forward_features(self, images: torch.Tensor, unflatten_patches: bool=False) -> Dict[str, torch.Tensor]:
        """
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W).
            unflatten_patches (bool): Whether the patch embeddings are returned as a contiguous sequence 
                                    of patches or a 2D spatial map. Default is flattened (False).
        
        Returns:
            output_dict (Dict[str, torch.Tensor]): convenient views of last hidden states for downstream use
                "x_norm": the last hidden state in full (including CLS and possible register or other tokens)
                "x_norm_clstoken": the CLS token for classification or semantic tasks
                "x_norm_patchtokens": the patch embeddings
        """
        # technically one should refrain from passing batches of shape B, S, C, H, W, but some
        # pipelines test models which take in both shape types 
        if len(images.shape) == 5:
            b, s, c, h, w = images.shape
            images = images.reshape(b * s, c, h, w)
            
        if self.preprocess_images:
            images = self.processor(images)

        _, _, h, w = images.shape
        num_patches_h, num_patches_w = h // self.patch_size, w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        last_hidden_states = self.model.forward_features(images) # (B, 1 + num_register_tokens + num_patches, feature_dim)

        patch_embeds = last_hidden_states[:, -num_patches:, :]
        if unflatten_patches:
            patch_embeds = patch_embeds.unflatten(1, (num_patches_h, num_patches_w))
        
        output_dict = {
            "x_norm": last_hidden_states,
            "x_norm_clstoken": last_hidden_states[:, 0],
            "x_norm_patchtokens": patch_embeds
        }
        return output_dict