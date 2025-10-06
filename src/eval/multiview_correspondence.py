import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from src.utils.camera import unproject, build_pose_matrix, invert_pose
from external.uco3d.uco3d import UCO3DDataset
from external.uco3d.uco3d.dataset_utils.utils import opencv_cameras_projection_from_uco3d

logger = logging.getLogger()

def downsample_world_coords(world_coords: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Downsample world_coords to match model's patch token resolution.
    
    Args:
        world_coords (torch.Tensor): (N, H, W, 3)
        patch_size (int): model patch size, e.g., 16

    Returns:
        torch.Tensor: (N, P_H, P_W, 3)
    """
    # permute for pooling
    coords = world_coords.permute(0, 3, 1, 2)

    # avg pool down to patch resolution
    # TODO: currently uses ceil_mode, which leads to edge bias. Probably need to create custom masked pooling with pre-padding
    coords_pooled = F.avg_pool2d(
        coords, 
        kernel_size=patch_size, 
        stride=patch_size,
        ceil_mode=True
    )
    return coords_pooled.permute(0, 2, 3, 1)

def ravel_hash_vec(arr: torch.Tensor) -> torch.Tensor:
    """
    Ravel the coordinates after subtracting the min coordinates.
    Code adapted from: https://github.com/Visual-AI/3DRS/blob/11ad004a9d81d7bdb0034cf19bdca457146e8892/llava/model/language_model/.ipynb_checkpoints/llava_qwen-checkpoint.py#L297 
    """
    a = arr.clone()
    a -= a.min(1, keepdims=True)[0]
    arr_max = a.max(1, keepdims=True)[0] + 1
    keys = torch.zeros(a.shape[0], a.shape[1], dtype=a.dtype, device=a.device)
    for j in range(a.shape[2] - 1):
        keys += a[..., j]
        keys *= arr_max[..., j + 1]
    keys += a[..., -1]
    return keys

def voxelize_world_coords(
    world_coords_pooled: torch.Tensor, 
    scene_point_cloud: Optional[torch.Tensor] = None,
    voxel_size: float=0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Discretize world points to voxel grid coordinates
    Code adapted from: https://github.com/Visual-AI/3DRS/blob/11ad004a9d81d7bdb0034cf19bdca457146e8892/llava/model/language_model/.ipynb_checkpoints/llava_qwen-checkpoint.py#L283

    Args:
        world_coords_pooled (torch.Tensor): Per patch pooled points in world coordinates (N, H_p, W_p, 3)
        scene_point_cloud (torch.Tensor, Optional): The scene point cloud to compute voxel boundaries (-1, 3)
        voxel_size (float): Size of each voxel

    Returns:
        tuple[torch.Tensor, torch.Tensor] of discretized world coordinates and their voxel ids
    """
    # TODO: dynamic voxel size based on box size

    if scene_point_cloud is not None:
        min_xyz, _ = scene_point_cloud.min(dim=0)
        max_xyz, _ = scene_point_cloud.max(dim=0)
    else:
        # min_xyz_range = [-15, -15, -5]
        # max_xyz_range = [15, 15, 5]
        logger.warning("Using dynamic coordinates can lead to noisier correspondence scores")
        min_xyz = world_coords_pooled.amin(dim=(0,1,2))
        max_xyz = world_coords_pooled.amax(dim=(0,1,2))

    min_xyz = torch.tensor(min_xyz).to(world_coords_pooled.device)
    max_xyz = torch.tensor(max_xyz).to(world_coords_pooled.device)

    clamped = torch.clamp(world_coords_pooled, min=min_xyz, max=max_xyz)

    # Convert to voxel indices
    world_coords_discrete = ((clamped - min_xyz) / voxel_size).round()
    keys = ravel_hash_vec(world_coords_discrete.reshape(1, -1, 3))
    key_set = torch.unique(keys[0].long(), return_inverse=True)

    return world_coords_discrete, key_set[1]

def correspondence_score(features: torch.Tensor, voxel_ids: torch.Tensor, chunk_size: int = 512) -> tuple[float, float]:
    """
    Correspondence score computed in chunks to avoid building full LxL matrices.

    Args:
        features (torch.Tensor): (N, H_p, W_p, D)
        voxel_ids (torch.Tensor): (L,) - long or int tensor mapping each feature to a voxel id
        chunk_size (int): number of rows to process at once (tune to memory limits)
    Returns:
        (corr_mean, non_corr_mean)
    """
    # Flatten features to (L, D)
    N, H_p, W_p, D = features.shape
    features_per_frame = H_p * W_p
    L = N * features_per_frame

    features = features.reshape(L, D)
    features = F.normalize(features, p=2, dim=-1)

    device = features.device
    if voxel_ids.device != device:
        voxel_ids = voxel_ids.to(device)

    # frame ids for checking cross-frame pairs
    frame_ids = torch.arange(L, device=device) // features_per_frame

    corr_sum = 0.0
    corr_count = 0
    noncorr_sum = 0.0
    noncorr_count = 0

    # Iterate over row chunks to compute similarities to all columns
    with torch.no_grad():
        # Precompute column indices for upper-tri check
        col_idx = torch.arange(L, device=device)

        for i_start in range(0, L, chunk_size):
            i_end = min(L, i_start + chunk_size)
            rows = features[i_start:i_end]  # (chunk, D)
            # compute similarities to all features: (chunk, L)
            sims = torch.matmul(rows, features.t())

            # Prepare row-wise scalars for masks
            row_idx = torch.arange(i_start, i_end, device=device)  # global row indices
            row_vox = voxel_ids[row_idx].unsqueeze(1)  # (chunk,1)
            # same voxel mask against all columns: (chunk, L)
            same_vox_mask = (row_vox == voxel_ids.unsqueeze(0))

            row_frames = frame_ids[row_idx].unsqueeze(1)  # (chunk,1)
            diff_frame_mask = (row_frames != frame_ids.unsqueeze(0))

            # upper triangular condition j > i (avoid double-count)
            upper_mask = (col_idx.unsqueeze(0) > row_idx.unsqueeze(1))  # (chunk, L)

            # correspondence masks
            corr_mask = same_vox_mask & diff_frame_mask & upper_mask
            noncorr_mask = (~same_vox_mask) & diff_frame_mask & upper_mask

            # Sum sims and counts for corr
            if corr_mask.any():
                s_sum = sims[corr_mask].sum().item()
                s_count = int(corr_mask.sum().item())
                corr_sum += s_sum
                corr_count += s_count

            # Sum sims and counts for non-corr
            if noncorr_mask.any():
                ns_sum = sims[noncorr_mask].sum().item()
                ns_count = int(noncorr_mask.sum().item())
                noncorr_sum += ns_sum
                noncorr_count += ns_count

            # free up memory
            del sims, same_vox_mask, diff_frame_mask, upper_mask, corr_mask, noncorr_mask

    # If no pairs found, avoid division by zero
    corr_mean = corr_sum / corr_count if corr_count > 0 else 0.0
    noncorr_mean = noncorr_sum / noncorr_count if noncorr_count > 0 else 0.0

    return float(corr_mean), float(noncorr_mean)


# TODO: BIG ONE: make this generalize to all datasets. Likely need to create wrapper class with methods that allows
# you to extract camera, depth, image, etc
def compute_correspondence_score(
    dataset: UCO3DDataset,
    model: nn.Module,   #TODO a FeatureExtractor class
    device: str | torch.device='cpu',
    resize_size: Optional[list[int] | torch.Tensor] = None
) -> Dict[str, Dict[str, float]]:
    scores = {}
    
    # TODO:
    #   1) Make this a dataset class instance
    #   2) Use path read from config
    num_frames_sampled = 10   # evenly spaced
    with open("/cluster/home/tsiebert/3DFeatProbe/test/debug-video-lengths.json") as f:
        num_frames_by_sequence = json.load(f)

    if resize_size is not None:
        if not torch.is_tensor(resize_size):
            resize_size = torch.LongTensor(resize_size)
        elif resize_size.dtype != torch.long:
            resize_size = resize_size.long()

    for sequence_name in dataset.sequence_names():
        logger.info(f"Processing sequence: {sequence_name}")
        # TODO:
        # co3d gives camera_quality_score and point_cloud_quality_score.
        # might want to filter frames with low scores before unprojecting, to avoid garbage world coords.
        # TODO: 
        # depth maps might help remove invalid points, but keeping track of the mask needs some thought
        # also need to handle 
        seq_len = num_frames_by_sequence[sequence_name]
        frames_to_sample = torch.linspace(0, seq_len-1, num_frames_sampled).long().tolist()

        views, poses, intrinsics = [], [], []
        depth_maps, depth_masks = [], []
        poses_for_wc = []
        for frame_idx in frames_to_sample:
            frame_data = dataset[sequence_name, frame_idx]
            if resize_size is not None:
                frame_data.resize_frame_(resize_size)

            image = frame_data.image_rgb    # Tensor of size 3, 800, 800 in [0,1]
            camera = frame_data.camera

            _, H, W = image.shape
            R, tvec, K = opencv_cameras_projection_from_uco3d(camera, torch.tensor([[H,W]]))
            pose = build_pose_matrix(R, tvec)

            depth_map = frame_data.depth_map    # Tensor of size 1, 800, 800

            views.append(image)
            poses.append(pose)
            intrinsics.append(K)

            depth_maps.append(depth_map)

            pose_for_wc = invert_pose(pose) # cam2world
            poses_for_wc.append(pose_for_wc)
            
        world_coords = unproject(
            torch.stack(intrinsics, dim=0).squeeze().to(device),
            torch.stack(poses_for_wc, dim=0).to(device), 
            torch.stack(depth_maps, dim=0).squeeze().to(device)
        )

        world_coords = downsample_world_coords(world_coords, patch_size=model.patch_size)
        pcd = dataset[sequence_name, 0].sequence_point_cloud.xyz

        _, voxel_ids = voxelize_world_coords(world_coords, scene_point_cloud=pcd, voxel_size=0.1)

        with torch.no_grad():
            features = model(torch.stack(views, dim=0).to(device))

        corr_score, non_corr_score = correspondence_score(features, voxel_ids, chunk_size=2048)
        scores[sequence_name] = {
            "corr": corr_score,
            "non_corr": non_corr_score
        }

    return scores