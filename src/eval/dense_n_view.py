# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Script to benchmark the dense multi-view metrics
"""
import json
import logging
import os
import sys
import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import warnings
from pathlib import Path
from typing import Dict, Callable
from omegaconf import DictConfig, OmegaConf
from mapanything.datasets import get_test_data_loader
from mapanything.utils.geometry import (
    geotrf,
    inv,
    normalize_multiple_pointclouds,
    quaternion_to_rotation_matrix,
    transform_pose_using_quats_and_trans_2_to_1,
)
from mapanything.utils.misc import StreamToLogger

from src.datasets import build_wai_dataloader
from src.training.utils import convert_mapa_batch_to_vggt, move_data_to_device
from src.models.vggt import VGGT
from src.utils.logging import direct_logger_to_stdout
from .metrics import *

log = logging.getLogger(__name__)
direct_logger_to_stdout(log)

# Metric Registry
METRIC_REGISTRY: Dict[str, Callable] = {}

def register_metric(name: str, metric_fn: Callable):
    """Register a metric function"""
    METRIC_REGISTRY[name] = metric_fn

register_metric("pointmaps_abs_rel", compute_pointmaps_abs_rel)
register_metric("pointmaps_inlier_thres_103", compute_pointmaps_inlier_thres_103)
register_metric("pointmaps_inlier_thres_105", compute_pointmaps_inlier_thres_105)
register_metric("pose_ate_rmse", compute_pose_ate_rmse)
register_metric("pose_auc_5", compute_pose_auc_5)
register_metric("pose_auc_30", compute_pose_auc_30)
register_metric("z_depth_abs_rel", compute_z_depth_abs_rel)
register_metric("z_depth_inlier_thres_103", compute_z_depth_inlier_thres_103)
register_metric("z_depth_inlier_thres_105", compute_z_depth_inlier_thres_105)
register_metric("ray_dirs_err_deg", compute_ray_dirs_err_deg)

# Utils
def get_info_for_metric_computation(batch, preds, norm_mode="avg_dis"):
    """
    Function to get all the information needed to compute the metrics.
    Returns all quantities normalized w.r.t. camera of view0.
    """
    n_views = len(batch)
    batch_size = batch[0]["camera_pose"].shape[0]

    # Everything is normalized w.r.t. camera of view0
    # Intialize lists to store data for all views
    # Ground truth quantities
    in_camera0 = inv(batch[0]["camera_pose"])
    no_norm_gt_pts = []
    no_norm_gt_pts3d_cam = []
    no_norm_gt_pose_trans = []
    valid_masks = []
    gt_ray_directions = []
    gt_pose_quats = []
    # Predicted quantities
    pred_camera0 = torch.eye(4, device=preds[0]["cam_quats"].device).unsqueeze(0)
    pred_camera0 = pred_camera0.repeat(batch_size, 1, 1)
    pred_camera0_rot = quaternion_to_rotation_matrix(preds[0]["cam_quats"].clone())
    pred_camera0[..., :3, :3] = pred_camera0_rot
    pred_camera0[..., :3, 3] = preds[0]["cam_trans"].clone()
    pred_in_camera0 = inv(pred_camera0)
    no_norm_pr_pts = []
    no_norm_pr_pts3d_cam = []
    no_norm_pr_pose_trans = []
    pr_ray_directions = []
    pr_pose_quats = []
    metric_pr_pts_to_compute_scale = []

    # Get ground truth & prediction info for all views
    for i in range(n_views):
        # Get the ground truth
        no_norm_gt_pts.append(geotrf(in_camera0, batch[i]["pts3d"]))
        valid_masks.append(batch[i]["valid_mask"].clone())
        gt_ray_directions.append(batch[i]["ray_directions_cam"])
        no_norm_gt_pts3d_cam.append(batch[i]["pts3d_cam"])
        if i == 0:
            # For view0, initialize identity pose
            gt_pose_quats.append(
                torch.tensor(
                    [0, 0, 0, 1],
                    dtype=gt_ray_directions[0].dtype,
                    device=gt_ray_directions[0].device,
                )
                .unsqueeze(0)
                .repeat(gt_ray_directions[0].shape[0], 1)
            )
            no_norm_gt_pose_trans.append(
                torch.tensor(
                    [0, 0, 0],
                    dtype=gt_ray_directions[0].dtype,
                    device=gt_ray_directions[0].device,
                )
                .unsqueeze(0)
                .repeat(gt_ray_directions[0].shape[0], 1)
            )
        else:
            # For other views, transform pose to view0's frame
            gt_pose_quats_world = batch[i]["camera_pose_quats"]
            no_norm_gt_pose_trans_world = batch[i]["camera_pose_trans"]
            gt_pose_quats_in_view0, no_norm_gt_pose_trans_in_view0 = (
                transform_pose_using_quats_and_trans_2_to_1(
                    batch[0]["camera_pose_quats"],
                    batch[0]["camera_pose_trans"],
                    gt_pose_quats_world,
                    no_norm_gt_pose_trans_world,
                )
            )
            gt_pose_quats.append(gt_pose_quats_in_view0)
            no_norm_gt_pose_trans.append(no_norm_gt_pose_trans_in_view0)

        # Get predicted pose & global pointmaps in view0's frame
        pr_pose_quats_in_view0, pr_pose_trans_in_view0 = (
            transform_pose_using_quats_and_trans_2_to_1(
                preds[0]["cam_quats"],
                preds[0]["cam_trans"],
                preds[i]["cam_quats"],
                preds[i]["cam_trans"],
            )
        )
        pr_pts3d_in_view0 = geotrf(pred_in_camera0, preds[i]["pts3d"])

        # Get predictions
        if "metric_scaling_factor" in preds[i].keys():
            # Divide by the predicted metric scaling factor to get the raw predicted points, pts3d_cam, and pose_trans
            # This detaches the predicted metric scaling factor from the geometry
            curr_view_no_norm_pr_pts = pr_pts3d_in_view0 / preds[i][
                "metric_scaling_factor"
            ].unsqueeze(-1).unsqueeze(-1)
            curr_view_no_norm_pr_pts3d_cam = preds[i]["pts3d_cam"] / preds[i][
                "metric_scaling_factor"
            ].unsqueeze(-1).unsqueeze(-1)
            curr_view_no_norm_pr_pose_trans = (
                pr_pose_trans_in_view0 / preds[i]["metric_scaling_factor"]
            )
        else:
            curr_view_no_norm_pr_pts = pr_pts3d_in_view0
            curr_view_no_norm_pr_pts3d_cam = preds[i]["pts3d_cam"]
            curr_view_no_norm_pr_pose_trans = pr_pose_trans_in_view0

        no_norm_pr_pts.append(curr_view_no_norm_pr_pts)
        no_norm_pr_pts3d_cam.append(curr_view_no_norm_pr_pts3d_cam)
        no_norm_pr_pose_trans.append(curr_view_no_norm_pr_pose_trans)
        pr_ray_directions.append(preds[i]["ray_directions"])
        pr_pose_quats.append(pr_pose_quats_in_view0)

        # Get the predicted metric scale points
        if "metric_scaling_factor" in preds[i].keys():
            curr_view_metric_pr_pts_to_compute_scale = (
                curr_view_no_norm_pr_pts.detach()
                * preds[i]["metric_scaling_factor"].unsqueeze(-1).unsqueeze(-1)
            )
        else:
            curr_view_metric_pr_pts_to_compute_scale = curr_view_no_norm_pr_pts.clone()
        metric_pr_pts_to_compute_scale.append(curr_view_metric_pr_pts_to_compute_scale)

    # Initialize normalized tensors
    gt_pts = [torch.zeros_like(pts) for pts in no_norm_gt_pts]
    gt_pts3d_cam = [torch.zeros_like(pts_cam) for pts_cam in no_norm_gt_pts3d_cam]
    gt_pose_trans = [torch.zeros_like(trans) for trans in no_norm_gt_pose_trans]

    pr_pts = [torch.zeros_like(pts) for pts in no_norm_pr_pts]
    pr_pts3d_cam = [torch.zeros_like(pts_cam) for pts_cam in no_norm_pr_pts3d_cam]
    pr_pose_trans = [torch.zeros_like(trans) for trans in no_norm_pr_pose_trans]

    # Normalize the predicted points
    pr_normalization_output = normalize_multiple_pointclouds(
        no_norm_pr_pts,
        valid_masks,
        norm_mode,
        ret_factor=True,
    )
    pr_pts_norm = pr_normalization_output[:-1]
    pr_norm_factor = pr_normalization_output[-1]

    # Normalize the ground truth points
    gt_normalization_output = normalize_multiple_pointclouds(
        no_norm_gt_pts, valid_masks, norm_mode, ret_factor=True
    )
    gt_pts_norm = gt_normalization_output[:-1]
    gt_norm_factor = gt_normalization_output[-1]

    for i in range(n_views):
        # Assign the normalized predictions
        pr_pts[i] = pr_pts_norm[i]
        pr_pts3d_cam[i] = no_norm_pr_pts3d_cam[i] / pr_norm_factor
        pr_pose_trans[i] = no_norm_pr_pose_trans[i] / pr_norm_factor[:, :, 0, 0]
        # Assign the normalized ground truth quantities
        gt_pts[i] = gt_pts_norm[i]
        gt_pts3d_cam[i] = no_norm_gt_pts3d_cam[i] / gt_norm_factor
        gt_pose_trans[i] = no_norm_gt_pose_trans[i] / gt_norm_factor[:, :, 0, 0]

    # Get the mask indicating ground truth metric scale quantities
    '''
    metric_scale_mask = batch[0]["is_metric_scale"]
    valid_gt_norm_factor_mask = (
        gt_norm_factor[:, 0, 0, 0] > 1e-8
    )  # Mask out cases where depth for all views is invalid
    valid_metric_scale_mask = metric_scale_mask & valid_gt_norm_factor_mask

    if valid_metric_scale_mask.any():
        # Compute the scale norm factor using the predicted metric scale points
        metric_pr_normalization_output = normalize_multiple_pointclouds(
            metric_pr_pts_to_compute_scale, valid_masks, norm_mode, ret_factor=True
        )
        pr_metric_norm_factor = metric_pr_normalization_output[-1]

        # Get the valid ground truth and predicted scale norm factors for the metric ground truth quantities
        gt_metric_norm_factor = gt_norm_factor[valid_metric_scale_mask]
        pr_metric_norm_factor = pr_metric_norm_factor[valid_metric_scale_mask]

        # Convert the ground truth and predicted scale norm factors to cpu
        gt_metric_norm_factor = gt_metric_norm_factor.cpu()
        pr_metric_norm_factor = pr_metric_norm_factor.cpu()
    else:
        gt_metric_norm_factor = None
        pr_metric_norm_factor = None
    '''

    # Convert the quaternions and translations to poses (4x4 matrices)
    # Convert the ray depth to z-depth
    # Also move the other quantites to CPU (keep poses on GPU)
    gt_poses = []
    gt_z_depths = []
    pr_poses = []
    pr_z_depths = []
    for i in range(n_views):
        # Get the ground truth pose
        gt_pose_curr_view = torch.eye(4, device=gt_pose_quats[i].device)
        gt_pose_curr_view = gt_pose_curr_view.repeat(gt_pose_quats[i].shape[0], 1, 1)
        gt_rot_curr_view = quaternion_to_rotation_matrix(gt_pose_quats[i])
        gt_pose_curr_view[..., :3, :3] = gt_rot_curr_view
        gt_pose_curr_view[..., :3, 3] = gt_pose_trans[i]
        gt_poses.append(gt_pose_curr_view)

        # Get the predicted pose
        pr_pose_curr_view = torch.eye(4, device=pr_pose_quats[i].device).unsqueeze(0)
        pr_pose_curr_view = pr_pose_curr_view.repeat(pr_pose_quats[i].shape[0], 1, 1)
        pr_rot_curr_view = quaternion_to_rotation_matrix(pr_pose_quats[i])
        pr_pose_curr_view[..., :3, :3] = pr_rot_curr_view
        pr_pose_curr_view[..., :3, 3] = pr_pose_trans[i]
        pr_poses.append(pr_pose_curr_view)

        # Get the ground truth z-depth
        gt_local_pts3d = gt_pts3d_cam[i]
        gt_z_depths.append(gt_local_pts3d[..., 2:].cpu())

        # Get the predicted z-depth
        pr_local_pts3d = pr_pts3d_cam[i]
        pr_z_depths.append(pr_local_pts3d[..., 2:].cpu())

        # Convert the other quantities to cpu
        gt_pts[i] = gt_pts[i].cpu()
        pr_pts[i] = pr_pts[i].cpu()
        valid_masks[i] = valid_masks[i].cpu()

    # Pack the required information into a dictionary
    gt_info = {
        "ray_directions": gt_ray_directions,
        "z_depths": gt_z_depths,
        "poses": gt_poses,
        "pts3d": gt_pts,
        # "metric_scale": gt_metric_norm_factor,
    }
    pr_info = {
        "ray_directions": pr_ray_directions,
        "z_depths": pr_z_depths,
        "poses": pr_poses,
        "pts3d": pr_pts,
        # "metric_scale": pr_metric_norm_factor,
    }

    return gt_info, pr_info, valid_masks

def build_dataset(dataset, batch_size, num_workers):
    """
    Builds data loaders for testing.

    Args:
        dataset: Dataset specification string.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.

    Returns:
        DataLoader: PyTorch DataLoader configured for the specified dataset.
    """
    log.info("Building data loader for dataset: ", dataset)
    loader = get_test_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=False,
        drop_last=False,
    )

    log.info("Dataset length: ", len(loader))
    return loader

# Main
def compute_results_for_batcb(
    batch, 
    preds, 
    metric_names,
    per_scene_results=None, 
    reduce_mean=False
):
    """
    Compute results for a batch 
    
    Args:
        batch: Input batch
        preds: Model output for the batch
        metric_names: List of metrics names to eval
        per_scene_results: A dict for per-scene results. Modified in place if not none
        reduce_mean: If True, scores are averaged across the batch
    Returns:
        Scores for each metric
    """
    # Get all the information needed to compute the metrics
    gt_info, pr_info, valid_masks = get_info_for_metric_computation(batch, preds)

    # Loop over each seq in the batch and compute the metrics across all views
    n_views = len(batch)
    batch_size = batch[0]["img"].shape[0]

    scores = {metric_name: 0.0 for metric_name in metric_names}

    for batch_idx in range(batch_size):
        # Get the scene of the multi-view set
        scene = batch[0]["label"][batch_idx]

        for metric_name in metric_names:
            if metric_name not in METRIC_REGISTRY:
                log.warning(f"Unknown metric: {metric_name}. Available: {list(METRIC_REGISTRY.keys())}. Skipping")
                continue
        
            metric = METRIC_REGISTRY[metric_name]
            score = metric(gt_info, pr_info, valid_masks, n_views, batch_idx)
            if isinstance(score, (torch.Tensor, np.ndarray)):
                score = float(score.item() if hasattr(score, 'item') else float(score))
            else:
                score = float(score)

            scores[metric_name] += score

            # Update per-scene-results
            if per_scene_results is not None:
                per_scene_results[scene][metric_name].append(score)
    
    if reduce_mean:
        scores = {k: v / batch_size for k, v in scores.items()}
    
    return scores

@torch.no_grad()
def benchmark(args):
    # Setup output dir
    log.info("Output Directory: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    log.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    log.info("{}".format(args).replace(", ", ",\n"))

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # CUDNN benchmark
    cudnn.benchmark = not args.disable_cudnn_benchmark

    # Determine the mixed precision floating point type
    if args.amp.enabled:
        if args.amp.amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif args.amp.amp_dtype == "bf16":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                warnings.warn(
                    "bf16 is not supported on this device. Using fp16 instead."
                )
                amp_dtype = torch.float16
        elif args.amp.amp_dtype == "fp32":
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    # Init Test Datasets and Dataloaders
    log.info("Building test dataset {:s}".format(args.dataset.test_dataset))
    data_loaders = {
        dataset.split("(")[0]: build_wai_dataloader(
            dataset=dataset, num_workers=args.dataset.num_workers, test=True, 
            multi_res=False, batch_size=args.batch_size,
        )
        for dataset in args.dataset.test_dataset.split("+")
        if "(" in dataset
    }

    # Load Model
    model_cfg = args.model
    if model_cfg.load_pretrained:
        model = VGGT.from_pretrained("facebook/VGGT-1B")
        model_name = "vggt-1b-pretrained"
        log.info("Initialized Model from `facebook/VGGT-1B'")
    else:
        model_args = model_cfg.model_config
        model = VGGT(**model_args)
        patch_embed_name = model_args.patch_embed_config.model_config.checkpoint_path.split('/')[-1].split('.')[0] \
            if model_args.patch_embed_config.model_config.checkpoint_path \
            else model_args.patch_embed_config.model_id.split('/')[1]
        model_name = f"vggt-{patch_embed_name}"
        log.info(f"Initialized Model: {str(model)}")
    model.to(device)  # Move model to device
    model.eval()

    ckpt_path = args.checkpoint_path
    if ckpt_path:
        log.info(f"Loading checkpoint from: '{ckpt_path}'")
        with open(ckpt_path, "rb") as f:
            ckpt = torch.load(f, map_location=device, weights_only=False)
        log.info(
            model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
        )
    
    # Create dictionary to keep track of the results across different benchmarking datasets
    per_dataset_results = {}
    eval_metrics = args.eval_metrics

    # Loop over the benchmarking datasets
    for benchmark_dataset_name, data_loader in data_loaders.items():
        benchmark_dataset_name = benchmark_dataset_name.replace(' ','')
        log.info(f"Benchmarking dataset: {benchmark_dataset_name}")
        data_loader.dataset.set_epoch(0)

        # Create dictionary to keep track of the results across different scenes
        per_scene_results = {}

        # Init list of metrics for each scene
        for dataset_scene in data_loader.dataset.dataset.scenes:
            per_scene_results[dataset_scene] = {
                metric_name: [] for metric_name in eval_metrics if metric_name in METRIC_REGISTRY
            }

        # Loop over the batches
        for batch in data_loader:
            # Convert batch to B, S, C, H, W and move to device
            model_ready_batch = convert_mapa_batch_to_vggt(batch) 
            model_ready_batch = move_data_to_device(model_ready_batch, device)

            # Run model inference
            # Length of preds is equal to the number of views
            with torch.autocast("cuda", enabled=bool(args.amp.enabled), dtype=amp_dtype):
                preds = model(model_ready_batch["images"])
                preds = model.convert_preds_to_mapa(preds)

            ### Compute results for batch
            for view in batch:
                view["idx"] = view["idx"][2:]
            ignore_keys = set([
                "depthmap",
                "dataset",
                "label",
                "instance",
                "idx",
                "true_shape",
                "rng",
                "data_norm_type",
            ])
            for view in batch:
                for name in view.keys():  # pseudo_focal
                    if name in ignore_keys:
                        continue
                    view[name] = view[name].to(device, non_blocking=True)
            compute_results_for_batcb(batch, preds, eval_metrics, per_scene_results)

        # Save per-scene results
        # scene -> metric -> list of scores for each multiview set sampled from scene
        with open(os.path.join(
            args.output_dir, f"{model_name}_{benchmark_dataset_name}_views-{args.dataset.num_views}.json"
        ), "w") as f:
            json.dump(per_scene_results, f, indent=4)

        # Aggregate results across scenes in dataset
        # metric -> avg across all scenes of that metric
        across_dataset_results = {}
        for scene in per_scene_results.keys():
            for metric in per_scene_results[scene].keys():
                if metric not in across_dataset_results.keys():
                    across_dataset_results[metric] = []
                across_dataset_results[metric].extend(per_scene_results[scene][metric])

        for metric in across_dataset_results.keys():
            across_dataset_results[metric] = np.mean(
                across_dataset_results[metric]
            ).item()

        '''
        # NOTE: disabled since the per-dataset results store this info
        # Save the average results across all scenes to a json file
        with open(
            os.path.join(
                args.output_dir, f"{benchmark_dataset_name}_avg_across_all_scenes.json"
            ),
            "w",
        ) as f:
            json.dump(across_dataset_results, f, indent=4)
        '''

        per_dataset_results[benchmark_dataset_name] = across_dataset_results

        # Print the average results across all scenes
        log.info(f"Avg Results for dataset {benchmark_dataset_name}:")
        log.info(json.dumps(across_dataset_results, indent=4))

    # Compute the average results across all datasets and add an average entry to the per dataset result dictionary
    # dataset (incl. overall) -> metric -> avg across all datasets of that metric
    average_results = {}
    for metric in per_dataset_results[next(iter(per_dataset_results))].keys():
        metric_values = [
            per_dataset_results[dataset][metric] for dataset in per_dataset_results
        ]
        average_results[metric] = np.mean(metric_values).item()
    per_dataset_results["overall"] = average_results

    # Save the per-dataset results + overall avg
    with open(os.path.join(args.output_dir, f"{model_name}_overall_views-{args.dataset.num_views}.json"), "w") as f:
        json.dump(per_dataset_results, f, indent=4)
    
    # Log the average across all datasets
    log.info(f"Overall Results across datasets:")
    log.info(json.dumps(average_results, indent=4))

@hydra.main(
    version_base=None, config_path="../../configs", config_name="dense_n_view_benchmark"
)
def execute_benchmarking(cfg: DictConfig):
    # Allow the config to be editable
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    # Redirect stdout and stderr to the logger
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

    # Run the testing
    benchmark(cfg)

if __name__ == "__main__":
    execute_benchmarking()  # noqa