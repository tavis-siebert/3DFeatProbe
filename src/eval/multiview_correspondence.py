import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from src.utils.camera import unproject, build_pose_matrix, invert_pose
from uco3d import UCO3DDataset
from uco3d.dataset_utils.utils import opencv_cameras_projection_from_uco3d

logger = logging.getLogger()

## DEBUG ##
import random
import numpy as np

def evaluate_pairs_and_precision(features, voxel_ids, sample_pairs=50000, ks=(1,5,10)):
    device = features.device
    N, H_p, W_p, D = features.shape
    features_per_frame = H_p * W_p
    L = N * features_per_frame
    features = F.normalize(features.reshape(L, D), p=2, dim=-1)

    frame_ids = torch.arange(L, device=device) // features_per_frame

    # Build index lists
    idx = torch.arange(L, device=device)
    # valid indices (exclude voxel_id == -1 if you set sentinel)
    valid_mask_idx = (voxel_ids >= 0)
    valid_idx = idx[valid_mask_idx]
    if len(valid_idx) == 0:
        raise RuntimeError("No valid patches")

    # Precompute pair masks (but we will sample)
    # Sample corr pairs (same voxel, different frame)
    same_vox = {}
    inv_map = {}
    # indices per voxel
    for i, vid in enumerate(voxel_ids.tolist()):
        if vid < 0: 
            continue
        inv_map.setdefault(int(vid), []).append(i)
    corr_pairs = []
    noncorr_pairs = []
    # build corr list
    for vid, inds in inv_map.items():
        # group per frame
        by_frame = {}
        for ii in inds:
            f = int(frame_ids[ii].item())
            by_frame.setdefault(f, []).append(ii)
        frames = list(by_frame.keys())
        if len(frames) < 2: 
            continue
        # sample cross-frame combos
        for i_frame in range(len(frames)):
            for j_frame in range(i_frame+1, len(frames)):
                A = by_frame[frames[i_frame]]
                B = by_frame[frames[j_frame]]
                for a in A:
                    for b in B:
                        corr_pairs.append((a,b))
    # build noncorr pairs by sampling random pairs from different voxels
    # naive but ok for sample limit
    all_valid = [i for i in range(L) if (voxel_ids[i] >= 0)]
    while len(noncorr_pairs) < sample_pairs and len(noncorr_pairs) < 10_000_000:
        a = random.choice(all_valid)
        b = random.choice(all_valid)
        if frame_ids[a] == frame_ids[b]:
            continue
        if voxel_ids[a] != voxel_ids[b]:
            noncorr_pairs.append((a,b))
    # sample arrays
    corr_pairs_sample = random.sample(corr_pairs, min(len(corr_pairs), sample_pairs))
    noncorr_pairs_sample = random.sample(noncorr_pairs, min(len(noncorr_pairs), sample_pairs))

    def sims_for(pairs):
        a_idx = torch.tensor([p[0] for p in pairs], device=device)
        b_idx = torch.tensor([p[1] for p in pairs], device=device)
        sims = (features[a_idx] * features[b_idx]).sum(dim=-1)
        return sims.cpu().numpy()

    sims_corr = sims_for(corr_pairs_sample)
    sims_noncorr = sims_for(noncorr_pairs_sample)

    # stats
    def stats(arr):
        return {"count": len(arr), "median": float(np.median(arr)), "90p": float(np.percentile(arr, 90))}
    stats_corr = stats(sims_corr)
    stats_noncorr = stats(sims_noncorr)

    # precision@k
    # For each anchor, compute top-k neighbors across other frames and check if any share voxel id
    # We'll sample anchors to speed
    anchors = random.sample(list(all_valid), min(5000, len(all_valid)))
    precs = {k: 0 for k in ks}
    for a in anchors:
        sim_row = (features[a].unsqueeze(0) @ features.t()).squeeze(0)  # (L,)
        # mask same frame and invalids
        sim_row[frame_ids == frame_ids[a]] = -999
        # get top max_k
        topk = torch.topk(sim_row, max(ks)).indices.cpu().tolist()
        for k in ks:
            topk_k = topk[:k]
            hit = any(int(voxel_ids[t]) == int(voxel_ids[a]) for t in topk_k)
            precs[k] += int(hit)
    for k in ks:
        precs[k] = precs[k] / len(anchors)

    return {"corr_stats": stats_corr, "noncorr_stats": stats_noncorr, "precision_at_k": precs}
###########

def downsample_world_coords(
        world_coords: torch.Tensor, 
        patch_size: int,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Downsample world_coords to match model's patch token resolution.
    
    Args:
        world_coords (torch.Tensor): (N, H, W, 3) 
        patch_size (int): model patch size, e.g., 16
        mask (torch.Tensor): if not None, which points to mark for exclusion in downstream correspondence calc.
                            Masked patches will be assigned a NaN value and ignored in subsequent calculations.
                            Size (N, H, W)

    Returns:
        tuple[torch.Tensor]: The (N, P_H, P_W, 3) pooled coordinates and the mask of which coordinates are valid (N, P_H, P_W)
    """
    # permute for pooling
    coords = world_coords.permute(0, 3, 1, 2)

    # avg pool down to patch resolution
    pooled_coords = F.avg_pool2d(
        coords, 
        kernel_size=patch_size, 
        stride=patch_size,
        ceil_mode=True
    )

    pooled_mask = None
    if mask is not None:
        m = mask.unsqueeze(1).float()
        pooled_mask = F.avg_pool2d(m, kernel_size=patch_size, stride=patch_size, ceil_mode=True)
        pooled_mask = (pooled_mask >= 0.5).squeeze(1)

    return pooled_coords.permute(0, 2, 3, 1), pooled_mask

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
    # TODO: this heavily biases to foreground (e.g. background patch points get clamped to object scene)
    if scene_point_cloud is not None:
        min_xyz, _ = scene_point_cloud.min(dim=0)
        max_xyz, _ = scene_point_cloud.max(dim=0)
    else:
        # min_xyz_range = [-15, -15, -5]
        # max_xyz_range = [15, 15, 5]
        logger.warning("Using dynamic coordinates can lead to noisier correspondence scores")
        min_xyz = world_coords_pooled.amin(dim=(0,1,2))
        max_xyz = world_coords_pooled.amax(dim=(0,1,2))

    min_xyz = torch.tensor(min_xyz).to(world_coords_pooled.device) * 2
    max_xyz = torch.tensor(max_xyz).to(world_coords_pooled.device) * 2

    clamped = torch.clamp(world_coords_pooled, min=min_xyz, max=max_xyz)

    # convert to voxel indices
    world_coords_discrete = ((clamped - min_xyz) / voxel_size).round()
    keys = ravel_hash_vec(world_coords_discrete.reshape(1, -1, 3))
    key_set = torch.unique(keys[0].long(), return_inverse=True)

    return world_coords_discrete, key_set[1]

def correspondence_score(features: torch.Tensor, voxel_ids: torch.Tensor, chunk_size: int = 1024) -> tuple[float, float]:
    """
    Correspondence score computed in chunks to avoid building full LxL matrices.

    Args:
        features (torch.Tensor): (N, H_p, W_p, D)
        voxel_ids (torch.Tensor): (L,) - long or int tensor mapping each feature to a voxel id
        chunk_size (int): number of rows to process at once (tune to memory limits)
    Returns:
        (corr_mean, non_corr_mean)
    """
    # flatten features to (L, D)
    N, H_p, W_p, D = features.shape
    features_per_frame = H_p * W_p
    L = N * features_per_frame

    features = F.normalize(features.reshape(L, D), p=2, dim=-1).half()

    device = features.device
    voxel_ids = voxel_ids.to(device)
    valid_mask = (voxel_ids != -1)
    
    # view ids for checking cross-frame pairs (i.e. assigns each patch to the view it came from)
    view_ids = torch.arange(L, device=device) // features_per_frame

    corr_sum = 0.0
    corr_count = 0
    noncorr_sum = 0.0
    noncorr_count = 0

    # Compute L x L similarity matrix (chunked over rows for memory reasons)
    with torch.no_grad():
        col_idx = torch.arange(L, device=device)
        for i_start in range(0, L, chunk_size):
            i_end = min(L, i_start + chunk_size)
            rows = features[i_start:i_end]  # (chunk, D)

            # compute cosine similarities between patchs: (chunk, L)
            sims = torch.matmul(rows, features.t()).float()

            # prepare row-wise scalars for masks
            row_idx = torch.arange(i_start, i_end, device=device)   # (chunk,)
            row_vox_ids = voxel_ids[row_idx].unsqueeze(1)           # (chunk, 1)
            row_view_ids = view_ids[row_idx].unsqueeze(1)           # (chunk, 1)
            row_valid_mask = valid_mask[row_idx].unsqueeze(1)       # (chunk, 1)

            same_vox_mask = (row_vox_ids == voxel_ids.unsqueeze(0))    # same voxels
            diff_views_mask = (row_view_ids != view_ids.unsqueeze(0))  # different views
            upper_triangle_mask = (col_idx.unsqueeze(0) > row_idx.unsqueeze(1))   # upper triangle (avoid dounle count)
            valid_patches_mask = row_valid_mask & valid_mask.unsqueeze(0)   # only valid points

            # correspondence masks (on-the-fly for memory efficiency)
            corr_mask = valid_patches_mask & same_vox_mask & diff_views_mask & upper_triangle_mask
            noncorr_mask = valid_patches_mask & (~same_vox_mask) & diff_views_mask & upper_triangle_mask
            # metrics needed for averaging
            if corr_mask.any():
                corr_sum += sims[corr_mask].sum().item()
                corr_count += int(corr_mask.sum().item())

            if noncorr_mask.any():
                noncorr_sum += sims[noncorr_mask].sum().item()
                noncorr_count += int(noncorr_mask.sum().item())

            del sims, corr_mask, noncorr_mask, valid_patches_mask, same_vox_mask, diff_views_mask, upper_triangle_mask

    # compute average correspondence and non-correspondence score (avoid division by zero)
    corr_mean = corr_sum / corr_count if corr_count > 0 else 0.0
    noncorr_mean = noncorr_sum / noncorr_count if noncorr_count > 0 else 0.0

    return float(corr_mean), float(noncorr_mean), corr_count, noncorr_count

# TODO: BIG ONE: make this generalize to all datasets. Likely need to create wrapper class with methods that allows
# you to extract camera, depth, image, etc
def compute_correspondence_score(
    dataset: UCO3DDataset,
    model: nn.Module,   #TODO a FeatureExtractor class
    device: str | torch.device='cpu',
    voxel_size: float=0.5
) -> Dict[str, Dict[str, float]]:
    scores = {}
    
    # TODO:
    #   1) Make this a dataset class instance
    #   2) Use path read from config
    num_frames_sampled = 10   # evenly spaced
    with open("/cluster/home/tsiebert/3DFeatProbe/test/debug-video-lengths.json") as f:
        num_frames_by_sequence = json.load(f)

    for sequence_name in dataset.sequence_names()[:5]:
        logger.info(f"Processing sequence: {sequence_name}")
        # TODO:
        # co3d gives camera_quality_score and point_cloud_quality_score.
        # might want to filter frames with low scores before unprojecting, to avoid garbage world coords.
        seq_len = num_frames_by_sequence[sequence_name]
        frames_to_sample = torch.linspace(0, seq_len-1, num_frames_sampled).long().tolist()

        views, poses, intrinsics = [], [], []
        depth_maps, depth_masks = [], []
        poses_for_wc = []
        for frame_idx in frames_to_sample:
            frame_data = dataset[sequence_name, frame_idx]

            image = frame_data.image_rgb
            camera = frame_data.camera

            _, H, W = image.shape
            R, tvec, K = opencv_cameras_projection_from_uco3d(camera, torch.tensor([[H,W]]))
            pose = build_pose_matrix(R, tvec)

            depth_map = frame_data.depth_map
            depth_mask = frame_data.depth_mask

            views.append(image)
            poses.append(pose)
            intrinsics.append(K)

            depth_maps.append(depth_map)
            depth_masks.append(depth_mask)

            pose_for_wc = invert_pose(pose) # cam2world
            poses_for_wc.append(pose_for_wc)
        
        # compute pointmap from depth and camera in world coordinates
        world_coords = unproject(
            torch.stack(intrinsics, dim=0).squeeze().to(device),
            torch.stack(poses_for_wc, dim=0).to(device), 
            torch.stack(depth_maps, dim=0).squeeze().to(device)
        )

        # pool world coordinates to assign one point per image patch
        patch_coords, valid_mask = downsample_world_coords(
            world_coords, patch_size=model.patch_size, mask=torch.stack(depth_masks, dim=0).squeeze()
        )
        pcd = dataset[sequence_name, 0].sequence_point_cloud.xyz
        
        # compute hypothetical voxels for each world point
        _, voxel_ids = voxelize_world_coords(patch_coords, scene_point_cloud=pcd, voxel_size=voxel_size)
        if valid_mask is not None:
            voxel_ids[~valid_mask.flatten()] = -1   # ignore invalid points 

        # Get model features
        with torch.no_grad():
            feats_out = model(torch.stack(views, dim=0).to(device))

        # Handle (potentially) multiple layers
        if isinstance(feats_out, torch.Tensor):
            feats_out = {"final": feats_out}  # assumes final layer
        elif isinstance(feats_out, (list, tuple)):
            feats_out = {f"layer_{i}": f for i, f in enumerate(feats_out)} 
        elif not isinstance(feats_out, dict):
            raise TypeError(f"Unexpected feature output type: {type(feats_out)}")

        # Compute scores per feature map
        layer_scores = {}
        for name, feats in feats_out.items():
            if len(feats.shape) == 5:
                feats = feats.squeeze(0)
            corr_score, noncorr_score, corr_count, noncorr_count = correspondence_score(feats, voxel_ids)
            layer_scores[name] = {
                "corr": corr_score,
                "non_corr": noncorr_score,
                "corr_count": corr_count,
                "noncorr_count": noncorr_count,
            }
            ## DEBUG ##
            print(
                evaluate_pairs_and_precision(feats, voxel_ids, )
            )
            ###########

            # memory handling
            del feats

        scores[sequence_name] = layer_scores

    return scores