import os
import json
from pathlib import Path
import hydra
import numpy as np
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

from src.training.utils import convert_mapa_batch_to_vggt, move_data_to_device
from src.datasets import build_wai_dataloader
from src.models import get_model_from_model_id

logger = logging.getLogger(__name__)

###########
## DEBUG ##
'''
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
'''
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
    # permute for pooling (N, 3, H, W)
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
    Compute correspondence and non-correspondence scores based on cosine similarity of features.
    Corresponding patches are defined as those that map to the same voxel in different views.

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

    features = F.normalize(features.reshape(L, D), p=2, dim=-1).half()

    device = features.device
    voxel_ids = voxel_ids.to(device)
    valid_mask = (voxel_ids != -1)
    
    # View ids for checking cross-frame pairs (i.e. assigns each patch to the view it came from)
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

@torch.no_grad()
def compute_correspondence_score(
    data_loader: torch.utils.data.DataLoader,
    model: nn.Module,   #TODO a FeatureExtractor class
    device: str,
    use_amp: bool=False,
    amp_dtype: torch.dtype=None,
    voxel_size: float=0.5
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Computes the correspondence and non-correspondence scores for a given model on a multiview dataset.

    Args:
        data_loader (torch.utils.data.DataLoader): WAI dataloader. To avoid errors, ensuren
                                                - num_views per sample ie > 1 and it is not variable
                                                - resolution is divisible by model patch size
        model (torch.nn.Module): the model to evaluate. Must return feature maps from forward pass
        device (str): device to run computations on
        use_amp (bool): whether to use automatic mixed precision during model inference
        amp_dtype (torch.dtype): data type to use for mixed precision
        voxel_size (float): size of each voxel in world coordinates. Unitless unless scene is defined using metric units
    
    Returns:
        the per_scene correspondence metrics of structure:
            {
                scene_name: {
                    "corr_score": {feature_map_name: [list of scores per sample]},
                    "noncorr_score": {feature_map_name: [list of scores per sample]},
                    "corr_count": {feature_map_name: [list of counts per sample]}, 
                    "noncorr_count": {feature_map_name: [list of counts per sample]},
                },
                ...
            }
    """
    per_scene_scores = {}
    # Init list of metrics for each scene
    for dataset_scene in data_loader.dataset.dataset.scenes:
        per_scene_scores[dataset_scene] = {
            "corr_score": {},
            "noncorr_score": {},
            "corr_count": {},
            "noncorr_count": {},
        }

    for batch in data_loader:
        # Gather features
        model_ready_batch = convert_mapa_batch_to_vggt(batch, world2cam=False)  # want cam2world for coorespondence
        model_ready_batch = move_data_to_device(model_ready_batch, device)
        with torch.autocast(device, enabled=use_amp, dtype=amp_dtype):
            feats_out = model(model_ready_batch["images"])
        
        # Aggregate (potentially) multiple layers
        if isinstance(feats_out, torch.Tensor):
            feats_out = {"final": feats_out}  # assumes final layer
        elif isinstance(feats_out, (list, tuple)):
            feats_out = {f"layer_{i}": f for i, f in enumerate(feats_out)} 
        elif not isinstance(feats_out, dict):
            raise TypeError(f"Unexpected feature output type: {type(feats_out)}")
        
        # Loop over each batch
        batch_size = batch["images"].shape[0]
        for batch_idx in range(batch_size):
            scene = batch[0]["label"][batch_idx]
            
            # Pool world coordinates to assign one point per image patch
            world_pts = model_ready_batch["world_points"][batch_idx]
            patch_coords, valid_mask = downsample_world_coords(
                world_coords=world_pts, patch_size=model.patch_size, mask=model_ready_batch["point_masks"][batch_idx]
            )
        
            # Compute hypothetical voxels for each world point
            _, voxel_ids = voxelize_world_coords(patch_coords, scene_point_cloud=world_pts, voxel_size=voxel_size)
            if valid_mask is not None:
                voxel_ids[~valid_mask.flatten()] = -1   # ignore invalid points

            # Compute scores per feature map
            per_scene_scores[scene] = {metric: {name: [] for name in feats_out.keys()} for metric in per_scene_scores[scene].keys()}

            for name, feats in feats_out.items():
                if len(feats.shape) == 5:
                    feats = feats[batch_idx]  # (num_views, H_p, W_p, D)

                corr_score, noncorr_score, corr_count, noncorr_count = correspondence_score(feats, voxel_ids)
                per_scene_scores[scene]["corr_score"][name].append(corr_score)
                per_scene_scores[scene]["noncorr_score"][name].append(noncorr_score)
                per_scene_scores[scene]["corr_count"][name].append(corr_count)
                per_scene_scores[scene]["noncorr_count"][name].append(noncorr_count)

                ###########
                ## DEBUG ##
                # print(evaluate_pairs_and_precision(feats, voxel_ids, ))
                ###########
    
    # Aggregate per-scene results
    for scene in per_scene_scores.keys():
        for metric in per_scene_scores[scene].keys():
            for layer_name in per_scene_scores[scene][metric].keys():
                per_scene_scores[scene][metric][layer_name] = np.mean(per_scene_scores[scene][metric][layer_name])

    return per_scene_scores

def benchmark(args):
    logging.info("Output Directory: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    # Determine the mixed precision floating point type
    if args.amp.enabled:
        if args.amp.amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif args.amp.amp_dtype == "bf16":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                logging.warning(
                    "bf16 is not supported on this device. Using fp16 instead."
                )
                amp_dtype = torch.float16
        elif args.amp.amp_dtype == "fp32":
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    # Build the test dataset(s)
    print("Building test dataset {:s}".format(args.dataset.test_dataset))
    data_loaders = {
        dataset.split("(")[0]: build_wai_dataloader(
            dataset=dataset, num_workers=args.dataset.num_workers, test=True, 
            multi_res=False, batch_size=args.batch_size,
        )
        for dataset in args.dataset.test_dataset.split("+")
        if "(" in dataset
    }

    # load model
    model = get_model_from_model_id(args.model.model_id, args.model.model_config)
    model.to(device)
    model.eval()
    #TODO: handle checkpoint loading

    # Run eval across datasets
    per_dataset_results = {}
    for benchmark_dataset_name, data_loader in data_loaders.items():
        logging.info("Benchmarking dataset: ", benchmark_dataset_name)
        data_loader.dataset.set_epoch(0)

        per_scene_results = compute_correspondence_score(
            dataset=data_loader, model=model, device=device, 
            use_amp=args.amp.enabled, amp_dtype=amp_dtype, voxel_size=args.voxel_size
        )

        # Aggregate results across all scenes
        cross_scene_results = {
            "corr_score": {},
            "noncorr_score": {},
            "corr_count": {},
            "noncorr_count": {},
        }
        for scene in per_scene_results.keys():
            for metric in per_scene_results[scene].keys():
                for layer_name in per_scene_results[scene][metric].keys():
                    cross_scene_results[metric].setdefault(layer_name, []).append(per_scene_results[scene][metric][layer_name])
        for metric in cross_scene_results.keys():
            for layer_name in cross_scene_results[metric].keys():
                cross_scene_results[metric][layer_name] = np.mean(cross_scene_results[metric][layer_name])

        # Save
        with open(
            os.path.join(
                args.output_dir, f"{benchmark_dataset_name}_avg_corr.json"
            ),
            "w",
        ) as f:
            json.dump(cross_scene_results, f, indent=4)
        
        per_dataset_results[benchmark_dataset_name] = cross_scene_results

        # Log
        logging.info(f"Results for dataset {benchmark_dataset_name}:")
        logging.info(json.dumps(cross_scene_results, indent=4))

    # Aggregate across datasets
    overall_results = {
        "corr_score": {},
        "noncorr_score": {},
        "corr_count": {},
        "noncorr_count": {},
    }
    for dataset_name in per_dataset_results.keys():
        for metric in per_dataset_results[dataset_name].keys():
            for layer_name in per_dataset_results[dataset_name][metric].keys():
                overall_results[metric].setdefault(layer_name, []).append(per_dataset_results[dataset_name][metric][layer_name])
    for metric in overall_results.keys():
        for layer_name in overall_results[metric].keys():
            overall_results[metric][layer_name] = np.mean(overall_results[metric][layer_name])

    # Save
    with open(
        os.path.join(
            args.output_dir, f"overall_avg_corr.json"
        ),
        "w",
    ) as f:
        json.dump(overall_results, f, indent=4)

    # Log
    logging.info(f"Overall Results across datasets:")
    logging.info(json.dumps(overall_results, indent=4))
