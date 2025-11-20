# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin 
import logging
from typing import Dict

from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    convert_z_depth_to_depth_along_ray,
    depthmap_to_camera_frame,
    get_rays_in_camera_frame,
)

from vggt.utils.rotation import mat_to_quat
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead

from src.utils.camera import *
from src.models.utils import interpolate_positional_embeddings

class VGGT(nn.Module, PyTorchModelHubMixin):
    """
    Wrapper class for VGGT (see the original project at: https://github.com/facebookresearch/vggt/tree/main)

    Added functionality 
        - Swapping in any DINOv2 checkpoint provided the modules / architecture are the same
        - Allows fine-grained control over Aggregator and Heads
    """
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        aggregator_config=None,
        camera_head_config=None,
        depth_head_config=None,
        point_head_config=None,
        track_head_config=None,
        **kwargs
    ):
        """
        Args:
            img_size (int): the image size to train on. Default = 518 (original size)
            patch_size (int): model patch size. Default = 14 (dinov2-large patch embed)
            embed_dim (int): model embed dim. Default = 1024 (dinov2-large patch embed)
            enable_camera (bool): if true, train camera head. Default = True
            enable_point (bool: if true, train pointmap head. Default = True
            enable_depth (bool): if true, train depthmap head. Default = True
            enable_track (bool): if true, train tracking head. Default = True
            aggregator_config (Dict, Optional): args to set up custom aggregator (e.g. change model depth, patch embed type).
                                      Default = None (only use original args)
            camera_head_config (Dict, Optional): args to set up custom camera head. Default = None
            depth_head_config (Dict, Optional): args to set up custom depth head. Default = None
            point_head_config (Dict, Optional): args to set up custom point head. Default = None
            track_head_config (Dict, Optional): args to set up custom track head. Default = None
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        agg_cfg = dict(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        if aggregator_config:
            agg_cfg.update(aggregator_config)

        self.aggregator = Aggregator(**agg_cfg)

        if enable_camera:
            cam_cfg = dict(dim_in=2 * embed_dim)
            if camera_head_config:
                cam_cfg.update(camera_head_config)
            self.camera_head = CameraHead(**cam_cfg)
        else:
            self.camera_head = None

        if enable_depth:
            depth_cfg = dict(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
            if depth_head_config:
                depth_cfg.update(depth_head_config)
            self.depth_head = DPTHead(**depth_cfg)
        else:
            self.depth_head = None

        if enable_point:
            point_cfg = dict(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
            if point_head_config:
                point_cfg.update(point_head_config)
            self.point_head = DPTHead(**point_cfg)
        else:
            self.point_head = None

        if enable_track:
            track_cfg = dict(dim_in=2 * embed_dim, patch_size=patch_size)
            if track_head_config:
                track_cfg.update(track_head_config)
            self.track_head = TrackHead(**track_cfg)
        else:
            self.track_head = None

    def load_pretrained_patch_embed(self, checkpoint_config: Dict):
        """
        Load pretrained DINOv2 weights into aggregator.patch_embed.

        Args:
            checkpont_config: (Dict): config for pretrained dinov2. 
                Must contain the following keys
                    - checkpoint_state_dict (Dict[str, torch.Tensor]): the state_dict of the checkpoint
                    - original_image_size (int, Tuple[int, int]): the size the original model was trained on
        
        Note:
            1. Ensure to trim any prefixes from `checkpoint_state_dict`
                (e.g. "module.submodule.pos_embed" -> "pos_embed")
            2. The embedding dim and register tokens must match
                (e.g. can't load model with dinov2-large and then load checkpoint from dinov2-base)
            3. The positional embedding will be interpolated to match the image size
        """
        patch_embed_state_dict = self.aggregator.patch_embed.state_dict()
        checkpoint_state_dict = checkpoint_config["checkpoint_state_dict"]

        # handle quirks with timm model
        if "reg_token" in checkpoint_state_dict:
            checkpoint_state_dict["register_tokens"] = checkpoint_state_dict.pop("reg_token")

        # handle positional embedding mismatch
        if "pos_embed" in checkpoint_state_dict:
            loaded_pos_embed, target_pos_embed = checkpoint_state_dict["pos_embed"], patch_embed_state_dict["pos_embed"]
            loaded_shape, target_shape = loaded_pos_embed.shape, target_pos_embed.shape

            # check embedding dims match
            if loaded_shape[-1] != target_shape[-1]:
                raise ValueError(f"Expected embed_dim={target_shape[-1]}. Got embed_dim={loaded_shape[-1]}. Ensure the checkpoint matches aggregator.patch_embed")
            
            # check patch dimension matches and interpolate otherwise
            if loaded_shape[1] != target_shape[1]:
                if target_shape[1] - loaded_shape[1] == 1:
                    logging.warning("Received no pos_embed for CLS token. Initializing to zeros. Dwonstream performance might degrade slightly")
                    checkpoint_state_dict["pos_embed"] = torch.cat((torch.zeros(1, 1, target_shape[-1]), loaded_pos_embed), dim=1)
                else:
                    original_image_size = checkpoint_config["original_image_size"]
                    if isinstance(original_image_size, int):
                        original_image_size = (original_image_size, original_image_size)
                    checkpoint_state_dict["pos_embed"] = interpolate_positional_embeddings(
                        loaded_pos_embed, original_image_size=original_image_size,
                        target_image_size=(self.img_size, self.img_size), patch_size=self.patch_size
                    )
        else:
            raise ValueError("Expected argument 'pos_embed' in `checkpoint_state_dict` but key doesn't exist")

        # load pretrained checkpoint
        missing, unexpected = self.aggregator.patch_embed.load_state_dict(checkpoint_state_dict, strict=False)
        if missing:
            if missing != ["mask_token"]:   # it's ok to not have masking
                raise ValueError(f"Missing keys: {missing}")
        if unexpected:
            logging.warning(f"Got unexpected keys: {unexpected}")
    
    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the VGGT model.
        Code adapted from: https://github.com/facebookresearch/vggt/blob/main/vggt/models/vggt.py

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.autocast(device_type=images.device.type, enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions
    
    def convert_preds_to_mapa(self, preds: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert model predictions to MAP-A format.

        Args:
            preds (Dict[str, torch.Tensor]): A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]

        Returns:
            List[dict]: A list containing the final predictions for all N views in mapanything format.
        """
        if not ("pose_enc" in preds and "depth" in preds):
            raise ValueError("Need at least camera and depth predictions to convert to map-anything format")
        _, num_views, H, W = preds["images"].shape

        pose_enc = preds["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, (H, W)
        )
        depth_map = preds["depth"]
        depth_conf = preds["depth_conf"] 

        res = []
        for view_idx in range(num_views):
            curr_view_extrinsic = extrinsic[:, view_idx, ...]
            curr_view_extrinsic = invert_pose(curr_view_extrinsic)
            curr_view_intrinsic = intrinsic[:, view_idx, ...]

            curr_view_depth_z = depth_map[:, view_idx, ...]
            curr_view_depth_z = curr_view_depth_z.squeeze(-1)
            curr_view_confidence = depth_conf[:, view_idx, ...]

            # get the camera frame pointmaps
            #TODO: if point head, use those points
            curr_view_pts3d_cam, _ = depthmap_to_camera_frame(
                curr_view_depth_z, curr_view_intrinsic
            )

            # convert the extrinsics to quaternions and translations
            curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
            curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])

            # Convert the z depth to depth along ray
            curr_view_depth_along_ray = convert_z_depth_to_depth_along_ray(
                curr_view_depth_z, curr_view_intrinsic
            )
            curr_view_depth_along_ray = curr_view_depth_along_ray.unsqueeze(-1)

            # Get the ray directions on the unit sphere in the camera frame
            _, curr_view_ray_dirs = get_rays_in_camera_frame(
                curr_view_intrinsic, H, W, normalize_to_unit_sphere=True
            )

            # Get the pointmaps
            curr_view_pts3d = (
                convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                    curr_view_ray_dirs,
                    curr_view_depth_along_ray,
                    curr_view_cam_translations,
                    curr_view_cam_quats,
                )
            )

            # Append the outputs to the result list
            res.append(
                {
                    "pts3d": curr_view_pts3d,
                    "pts3d_cam": curr_view_pts3d_cam,
                    "ray_directions": curr_view_ray_dirs,
                    "depth_along_ray": curr_view_depth_along_ray,
                    "cam_trans": curr_view_cam_translations,
                    "cam_quats": curr_view_cam_quats,
                    "conf": curr_view_confidence,
                }
            )
            
        return res

            