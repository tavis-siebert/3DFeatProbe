import os
import json
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from src.utils.logging import direct_logger_to_stdout
from src.training.utils import convert_mapa_batch_to_vggt, move_data_to_device
from src.datasets import build_wai_dataloader
from src.models import get_model_from_model_id
from src.models.feature_extractors import FeatureExtractor

# Logging
log = logging.getLogger(__name__)
direct_logger_to_stdout(log)

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
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Downsample world_coords to match model's patch token resolution.
    
    Args:
        world_coords (torch.Tensor): (N, H, W, 3) 
        patch_size (int): model patch size, e.g., 16
        valid_mask (torch.Tensor): if not None, which points to mark for exclusion in downstream
                            correspondence calc (1 = valid, 0 = exclude). Size (N, H, W)

    Returns:
        Tuple[torch.Tensor]: The (N, H_p, W_p, 3) pooled coordinates 
                            and the mask of which coordinates are valid (N, H_p, W_p)
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
    if valid_mask is not None:
        m = valid_mask.unsqueeze(1).float()
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
    valid_mask: Optional[torch.Tensor] = None,
    scene_point_cloud: Optional[torch.Tensor] = None,
    voxel_size: float = 0.1,
    padding_fraction: float=0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Discretize world points to voxel grid coordinates
    Code adapted from: https://github.com/Visual-AI/3DRS/blob/11ad004a9d81d7bdb0034cf19bdca457146e8892/llava/model/language_model/.ipynb_checkpoints/llava_qwen-checkpoint.py#L283

    Args:
        world_coords_pooled (torch.Tensor): Per patch pooled points in world coordinates (N, H_p, W_p, 3)
        valid_mask (torch.Tensor, Optional): Per patch mask of whether the corresponding pooled point is valid (mask == 1) (N, H_p, W_p)
        scene_point_cloud (torch.Tensor, Optional): The scene point cloud to compute voxel boundaries (-1, 3)
        voxel_size (float): Size of each voxel
        padding_fraction (float): Fraction of each dimension to add on as padding to voxel volume dimensions

    Returns:
        Tuple[torch.Tensor, torch.Tensor] of discretized world coordinates and their voxel ids
    """
    N, H_p, W_p, = world_coords_pooled.shape[:-1]
    coords_flat = world_coords_pooled.reshape(-1, 3)
        
    if scene_point_cloud is not None:
        min_xyz = scene_point_cloud.min(dim=0)[0]
        max_xyz = scene_point_cloud.max(dim=0)[0]
    elif valid_mask is not None:
        valid_coords = coords_flat[valid_mask.flatten()]
        min_xyz = valid_coords.min(dim=0)[0]
        max_xyz = valid_coords.max(dim=0)[0]
    else:
        min_xyz = coords_flat.min(dim=0)[0]
        max_xyz = coords_flat.max(dim=0)[0]

    # Pad symmetrically by fraction of range
    box_dims = (max_xyz - min_xyz).abs()
    pad = box_dims * float(padding_fraction)
    min_xyz = (min_xyz - pad)
    max_xyz = (max_xyz + pad)

    # Clamp, compute discrete coords
    coords_clamped = torch.clamp(coords_flat, min=min_xyz, max=max_xyz)
    world_coords_discrete = ((coords_clamped - min_xyz) / voxel_size).round()
    world_coords_discrete = world_coords_discrete.to(torch.long)

    # Hash to voxel IDs
    keys = ravel_hash_vec(world_coords_discrete.reshape(1, -1, 3))
    key_set = torch.unique(keys[0].long(), return_inverse=True)

    # Label invalid voxels with -1
    voxel_ids = key_set[1]
    if valid_mask is not None:
        voxel_ids[~valid_mask.flatten()] = -1

    return world_coords_discrete.reshape(N, H_p, W_p, 3), voxel_ids

def correspondence_score(features: torch.Tensor, voxel_ids: torch.Tensor, chunk_size: int = 1024) -> tuple[float, float]:
    """
    Compute correspondence and non-correspondence scores based on cosine similarity of features.
    Corresponding patches are defined as those that map to the same voxel in different views.

    Args:
        features (torch.Tensor): (N, P, D) or (N, H_p, W_p, D) where P = H_p * W_p
        voxel_ids (torch.Tensor): (L,) - long or int tensor mapping each feature to a voxel id
        chunk_size (int): number of rows to process at once (tune to memory limits)
    Returns:
        (corr_mean, non_corr_mean)
    """
    # Flatten features to (L, D)
    if len(features.shape) == 4:
        N, H_p, W_p, D = features.shape
        P = H_p * W_p
    else:
        N, P, D = features.shape
    L = N * P

    features = F.normalize(features.reshape(L, D), p=2, dim=-1).half()

    device = features.device
    voxel_ids = voxel_ids.to(device)
    valid_mask = (voxel_ids != -1)
    
    # View ids for checking cross-frame pairs (i.e. assigns each patch to the view it came from)
    view_ids = torch.arange(L, device=device) // P

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
    model: FeatureExtractor,
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
        # Convert batch to B, S, C, H, W and move to device
        model_ready_batch = convert_mapa_batch_to_vggt(batch, world2cam=False)  # want cam2world for coorespondence
        model_ready_batch = move_data_to_device(model_ready_batch, device)

        # Get features
        with torch.autocast(device, enabled=use_amp, dtype=amp_dtype):
            B, S = model_ready_batch["images"].shape[:2]
            feats_out = model.forward_features(model_ready_batch["images"])
        
        # Aggregate
        output_schema = model.validate_output_schema(feats_out)
        if output_schema == "single":
            D = feats_out["x_norm_patchtokens"].shape[-1]
            layerwise_feats = {"final": feats_out["x_norm_patchtokens"].reshape(B, S, -1, D)}
        elif output_schema == "multi":
            layerwise_feats = {}
            for layer_name, layer_out in feats_out.items():
                patchtokens = layer_out["x_norm_patchtokens"]
                D = patchtokens.shape[-1]
                layerwise_feats[layer_name] = patchtokens.reshape(B, S, -1, D)
        
        # Loop over each batch
        for batch_idx in range(B):
            scene = batch[0]["label"][batch_idx]
            
            # Pool world coordinates to assign one point per image patch
            world_pts = model_ready_batch["world_points"][batch_idx]
            patch_coords, valid_mask = downsample_world_coords(
                world_coords=world_pts, patch_size=model.patch_size, mask=model_ready_batch["point_masks"][batch_idx]
            )
        
            # Compute hypothetical voxels for each valid world point
            _, voxel_ids = voxelize_world_coords(patch_coords, valid_mask, voxel_size=voxel_size)

            # Compute scores per feature map
            per_scene_scores[scene] = {metric: {name: [] for name in layerwise_feats.keys()} for metric in per_scene_scores[scene].keys()}

            for name, feats in layerwise_feats.items():
                corr_score, noncorr_score, corr_count, noncorr_count = correspondence_score(feats[batch_idx], voxel_ids)
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
    log.info("Output Directory: " + args.output_dir)
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
                log.warning(
                    "bf16 is not supported on this device. Using fp16 instead."
                )
                amp_dtype = torch.float16
        elif args.amp.amp_dtype == "fp32":
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    # Build the test dataset(s)
    log.info("Building test dataset {:s}".format(args.dataset.test_dataset))
    data_loaders = {
        dataset.split("(")[0]: build_wai_dataloader(
            dataset=dataset, num_workers=args.dataset.num_workers, test=True, 
            multi_res=False, batch_size=args.batch_size,
        )
        for dataset in args.dataset.test_dataset.split("+")
        if "(" in dataset
    }

    # Load model
    assert "feature_extractor" in args.model.model_id, "Can only test correspondence on instances of FeatureExtractor"
    model = get_model_from_model_id(args.model.model_id, args.model.model_config)
    model.to(device)
    model.eval()
        
    # for output naming
    model_name = model.checkpoint_path.split('/')[-1].split('.')[0] if model.checkpoint_path else args.model.model_id.split('/')[1]

    # Run eval across datasets
    per_dataset_results = {}
    for benchmark_dataset_name, data_loader in data_loaders.items():
        benchmark_dataset_name = benchmark_dataset_name.replace(' ','')
        log.info(f"Benchmarking dataset: {benchmark_dataset_name}")
        data_loader.dataset.set_epoch(0)

        # Compute per-scene stats
        per_scene_results = compute_correspondence_score(
            data_loader=data_loader, model=model, device=device, 
            use_amp=args.amp.enabled, amp_dtype=amp_dtype, voxel_size=args.voxel_size
        )

        # Save per-scene results
        out_path = f"mvcorr_{model_name}_{benchmark_dataset_name}_vox-{args.voxel_size}.json"
        with open(os.path.join(args.output_dir, out_path), "w") as f:
            json.dump(per_scene_results, f, indent=4)

        # Aggregate results across all scenes in dataset
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
        
        per_dataset_results[benchmark_dataset_name] = cross_scene_results

        # Log average across dataset
        log.info(f"Avg Results for dataset {benchmark_dataset_name}:")
        log.info(json.dumps(cross_scene_results, indent=4))

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

    per_dataset_results["overall"] = overall_results

    # Save per-dataset results + overall avg
    out_path = f"mvcorr_overall_{model_name}_vox-{args.voxel_size}.json"
    with open(os.path.join(args.output_dir, out_path), "w") as f:
        json.dump(per_dataset_results, f, indent=4)

    # Log the average across all datasets
    log.info(f"Overall Results across datasets:")
    log.info(json.dumps(overall_results, indent=4))