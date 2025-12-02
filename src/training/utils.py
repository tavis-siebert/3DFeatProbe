# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import logging
import random
from wcmatch import fnmatch
from functools import wraps
from typing import List, Dict, Union, Optional, Tuple, Sequence, Any

from src.utils.camera import invert_pose

logger = logging.getLogger(__name__)

#-----------#
#  General  #
#-----------#
def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
    
    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                  [-hard_max, hard_max] will be clamped. If None, 
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor
    
    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor

def move_data_to_device(data: Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]], device: torch.device, **kwargs: Any):
    """
    Function that recursively copies data to a torch.device.
    Args:
        data (torch.Tensor or Sequence[torch.Tensor] or Dict[torch.Tensor]): a tensor or list/tuple/dict of tensors (technically further recursion works too)
        device (torch.device): the device to move tensors to
        kwargs: e.g., `pin_memory`, `non_blocking`
    """
    if torch.is_tensor(data):
        return data.to(device, **kwargs)
    elif isinstance(data, (list, tuple)):
        return type(data)(move_data_to_device(t, device, **kwargs) for t in data)
    elif isinstance(data, dict):
        return type(data)({
            k: move_data_to_device(v, device, **kwargs)
            for k, v in data.items()
        })
    else:
        raise ValueError(f"Type {type(data)} is unsupoorted. Must be a tensor or list, tuple, dict of tensors")

def set_seeds(seed_value, rank: int=0):
    """
    Set the python random, numpy and torch seed for each gpu
    """
    seed_value = seed_value + rank
    logger.info(f"GPU {rank} SEED: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

#-------------------#
#  Data Processing  #
#-------------------#
def convert_mapa_batch_to_vggt(views: List[Dict], world2cam: bool = True) -> Dict[str, torch.Tensor]:
    """
    Convert map-anything's list-of-view-dicts format to VGGT's batched format.
    More generally it converts the map-anything per-view structure to a batch 
    with B, num_views, ... shape for each modality
    
    Args:
        views: List of view dictionaries from map-anything dataloader
        world2cam (bool): If True, convert cam2world poses to world2cam extrinsics. Default = True (VGGT expects world2cam)
        
    Returns:
        expected batch for VGGT input and loss
    """ 
    def __convert_from_numpy(arr: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr)
        return arr

    image_batch, depth_batch, valid_mask_batch = [], [], []
    intrinsics_batch, extrinsics_batch = [], []
    world_pts_batch, cam_pts_batch = [], []

    for view in views:
        # rgb image
        image = view['img'] # [B, 3, H, W]
        image_batch.append(__convert_from_numpy(image))

        # depth map
        depthmap = view['depthmap']  # [B, H, W, 1]
        depth_batch.append(__convert_from_numpy(depthmap))

        # mask
        valid_mask = view['valid_mask']  # [B, H, W]
        valid_mask_batch.append(__convert_from_numpy(valid_mask))

        # camera intrinsics
        intrinsics = view['camera_intrinsics']  # [B, 3, 3]
        intrinsics_batch.append(__convert_from_numpy(intrinsics))

        # camera pose
        pose = __convert_from_numpy(view['camera_pose']) # [B, 4, 4]
        if world2cam:
            pose = invert_pose(pose)
        extrinsics = pose[:, :3, :]  # [B, 3, 4]
        extrinsics_batch.append(extrinsics)

        # point maps
        pts3d = view['pts3d']  # [B, H, W, 3]
        pts3d_cam = view["pts3d_cam"]
        world_pts_batch.append(__convert_from_numpy(pts3d))
        cam_pts_batch.append(__convert_from_numpy(pts3d_cam))

    # stack and arrange as (B, num_views, ...)
    image_batch = torch.stack(image_batch, dim=1)
    depth_batch = torch.stack(depth_batch, dim=1).squeeze()
    valid_mask_batch = torch.stack(valid_mask_batch, dim=1)
    intrinsics_batch = torch.stack(intrinsics_batch, dim=1)
    extrinsics_batch = torch.stack(extrinsics_batch, dim=1)
    world_pts_batch = torch.stack(world_pts_batch, dim=1)
    cam_pts_batch = torch.stack(cam_pts_batch, dim=1)
    
    return {
        "images": image_batch,
        "extrinsics": extrinsics_batch,
        "intrinsics": intrinsics_batch,
        "depths": depth_batch,
        "world_points": world_pts_batch,
        "cam_points": cam_pts_batch,
        "point_masks": valid_mask_batch,
    }

def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_points: bool = True,
    point_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
    """
    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    assert device == torch.device("cpu")

    # Convert extrinsics to homogeneous form: (B, N,4,4)
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = invert_pose(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)


    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None


    if scale_by_points:
        new_cam_points = cam_points.clone()
        new_depths = depths.clone()

        dist = new_world_points.norm(dim=-1)
        dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        valid_count = point_masks.sum(dim=[1,2,3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)


        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths

    new_extrinsics = new_extrinsics[:, :, :3] # 4x4 -> 3x4
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)


    return new_extrinsics, new_cam_points, new_world_points, new_depths

#------------------#
#  Gradient Accum  #
#------------------#
def chunk_batch_for_accum_steps(
    batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
    accum_steps: int,
) -> List[torch.Tensor]:
    """Returns a chunked version of a training batch if gradient accumulation is enabled"""
    def _chunk_data(
        data: Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]],
        chunk_id: int,
        num_chunks: int
    ):
        # TODO: the logic is the same as move_device_to_batch, so I can probably refactor with a tensor_op function arg
        if torch.is_tensor(data):
            start = (len(data) // num_chunks) * chunk_id
            end = (len(data) // num_chunks) * (chunk_id + 1)
            return data[start:end] 
        elif isinstance(data, (list, tuple)):
            return type(data)(_chunk_data(t, chunk_id, num_chunks) for t in data)
        elif isinstance(data, dict):
            return {
                k: _chunk_data(v, chunk_id, num_chunks)
                for k, v in data.items()
            }
        else:
            raise ValueError(f"Type {type(data)} is unsupoorted. Must be a tensor or list, tuple, dict of tensors")

    if accum_steps == 1:
        return [batch]
    return [_chunk_data(batch, chunk_id, accum_steps) for chunk_id in range(accum_steps)]
    
    
    

#--------------------#
#  Freezing Modules  #
#--------------------#
# Glob‑matching flags (behave like the Unix shell) 
GLOB_FLAGS = (
    fnmatch.CASE       # case‑sensitive
    | fnmatch.DOTMATCH # '*' also matches '.'
    | fnmatch.EXTMATCH # extended patterns like *(foo|bar)
    | fnmatch.SPLIT    # "pat1|pat2" works out‑of‑the‑box
)

def freeze_modules(model: nn.Module, patterns: List[str], recursive: bool = True) -> nn.Module:
    """Freeze (stop training) parts of *model* whose *name* matches *patterns*.

    Parameters
    ----------
    model : nn.Module
        The complete model you are working with.
    patterns : list[str]
        Glob patterns to match submodule names.  Example: ``["encoder.*", "cls_head"]``
    recursive : bool, default = True
        • ``True``  → also freeze every child of a matched module.
        • ``False`` → freeze only the matched module itself.

    Returns
    -------
    nn.Module
        The same model object, now with some parts frozen.

    Example
    -------
    >>> freeze_modules(model, ["encoder.*", "decoder.layer1"], recursive=True)
    """
    matched: set[str] = set()

    for name, mod in model.named_modules():
        # does *name* match ANY user pattern?
        if any(fnmatch.fnmatch(name, p, flags=GLOB_FLAGS) for p in patterns):
            matched.add(name)
            _freeze(mod, recursive)

    _check_every_pattern_used(matched, patterns)
    return model

def _freeze(mod: nn.Module, recursive: bool) -> None:
    """Put *mod* in eval mode and lock its parameters."""

    if recursive:
        mod.eval()            # affects the whole subtree
    else:
        mod.training = False  # only this exact module

    original_train = mod.train

    @wraps(original_train)
    def locked_train(mode: bool = True):
        if recursive:
            return original_train(False)  # ignore user's *mode*
        out = original_train(mode)        # children follow user's choice
        out.training = False              # but this module stays frozen
        return out

    mod.train = locked_train  # type: ignore[attr-defined]

    param_iter = (
        mod.parameters()              # default recurse=True
        if recursive
        else mod.parameters(recurse=False)
    )
    for p in param_iter:
        p.requires_grad = False


def _check_every_pattern_used(matched_names: set[str], patterns: List[str]):
    unused = [p for p in patterns if not any(fnmatch.fnmatch(n, p, flags=GLOB_FLAGS)
                                             for n in matched_names)]
    if unused:
        raise ValueError(f"These patterns matched nothing: {unused}")