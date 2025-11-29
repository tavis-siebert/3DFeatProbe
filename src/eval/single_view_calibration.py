# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Script to benchmark the image calibration performance
"""

import json
import logging
import sys
import os
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from mapanything.models import init_model
from mapanything.utils.metrics import (
    l2_distance_of_unit_ray_directions_to_angular_error,
)

from src.datasets import build_wai_dataloader
from src.models.vggt import VGGT

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
log.addHandler(handler)


def get_all_info_for_metric_computation(batch, preds):
    """
    Function to get all the information needed to compute the metrics.
    Batch must be in map-anything format
    """
    n_views = len(batch)

    # Intialize lists to store data for all views
    # Ground truth quantities
    gt_ray_directions = []
    # Predicted quantities
    pr_ray_directions = []

    # Get ground truth & prediction info for all views
    for i in range(n_views):
        gt_ray_directions.append(batch[i]["ray_directions_cam"])
        pr_ray_directions.append(preds[i]["ray_directions"])

    # Pack the required information into a dictionary
    gt_info = {
        "ray_directions": gt_ray_directions,
    }
    pr_info = {
        "ray_directions": pr_ray_directions,
    }

    return gt_info, pr_info

@torch.no_grad()
def benchmark(args):
    log.info("Output Directory: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    log.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    log.info("{}".format(args).replace(", ", ",\n"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

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
            multi_res=True, batch_size=args.batch_size, 
        )
        for dataset in args.dataset.test_dataset.split("+")
        if "(" in dataset
    }

    # Load Model
    '''
    #TODO: change to proper init_model function if not doing VGGT variants
    model = init_model(
        args.model.model_str, args.model.model_config, torch_hub_force_reload=False
    )
    '''
    model_cfg = args.model
    if model_cfg.load_pretrained:
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    else:
        model = VGGT(**model_cfg.model_config)
        
    model.to(device)  # Move model to device
    model.eval()

    checkpoint_path = args.checkpoint_path
    if checkpoint_path:
        log.info("Loading from checkpoint: ", checkpoint_path)
        with open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(
            model_state_dict, strict=False
        )
        if missing:
            raise ValueError(f"Missing keys: {missing}")
        if unexpected:
            logging.warning(f"Got unexpected keys: {unexpected}")
            
        del checkpoint

    # Create dictionary to keep track of the results across different benchmarking datasets
    per_dataset_results = {}

    # Loop over the benchmarking datasets
    for benchmark_dataset_name, data_loader in data_loaders.items():
        log.info("Benchmarking dataset: ", benchmark_dataset_name)
        data_loader.dataset.set_epoch(0)

        # Create dictionary to keep track of the results across different scenes
        per_scene_results = {}

        # Init list of metrics for each scene
        for dataset_scene in data_loader.dataset.dataset.scenes:
            per_scene_results[dataset_scene] = {
                "ray_dirs_err_deg": [],
            }

        # Loop over the batches
        for batch in data_loader:
            n_views = len(batch)
            # Remove unnecessary indices
            for view in batch:
                view["idx"] = view["idx"][2:]

            # Transfer batch to device
            ignore_keys = set(
                [
                    "depthmap",
                    "dataset",
                    "label",
                    "instance",
                    "idx",
                    "true_shape",
                    "rng",
                    "data_norm_type",
                ]
            )
            for view in batch:
                for name in view.keys():  # pseudo_focal
                    if name in ignore_keys:
                        continue
                    view[name] = view[name].to(device, non_blocking=True)

            # Run model inference
            img_list = [view["img"] for view in batch]
            images = torch.stack(img_list, dim=1)
            # length of preds is equal to the number of views
            with torch.autocast("cuda", enabled=bool(args.amp), dtype=amp_dtype):
                preds = model(images)
                preds = model.convert_preds_to_mapa(preds)

            # Get all the information needed to compute the metrics
            gt_info, pr_info = get_all_info_for_metric_computation(batch, preds)

            # Loop over each set in the batch and compute the metrics across all views
            batch_size = batch[0]["img"].shape[0]
            for batch_idx in range(batch_size):
                # Get the scene of the multi-view set
                scene = batch[0]["label"][batch_idx]

                # Compute the metrics across all views
                ray_dirs_err_deg_across_views = []
                for view_idx in range(n_views):
                    # Compute the l2 norm of the ray directions and convert it to angular error in degrees
                    ray_dirs_l2 = torch.norm(
                        gt_info["ray_directions"][view_idx][batch_idx]
                        - pr_info["ray_directions"][view_idx][batch_idx],
                        dim=-1,
                    )
                    ray_dirs_err_deg_curr_view = (
                        l2_distance_of_unit_ray_directions_to_angular_error(ray_dirs_l2)
                    )
                    ray_dirs_err_deg_curr_view = torch.mean(ray_dirs_err_deg_curr_view)
                    ray_dirs_err_deg_across_views.append(
                        ray_dirs_err_deg_curr_view.cpu().numpy()
                    )

                # Compute the average across all views
                ray_dirs_err_deg_curr_set = np.mean(ray_dirs_err_deg_across_views)

                # Append the metrics to the respective lists
                per_scene_results[scene]["ray_dirs_err_deg"].append(
                    ray_dirs_err_deg_curr_set.item()
                )

        # Save the per scene results to a json file
        with open(
            os.path.join(
                args.output_dir, f"{benchmark_dataset_name}_per_scene_results.json"
            ),
            "w",
        ) as f:
            json.dump(per_scene_results, f, indent=4)

        # Aggregate the per scene results
        across_dataset_results = {}
        for scene in per_scene_results.keys():
            for metric in per_scene_results[scene].keys():
                if metric not in across_dataset_results.keys():
                    across_dataset_results[metric] = []
                across_dataset_results[metric].extend(per_scene_results[scene][metric])

        # Compute the mean across all scenes
        for metric in across_dataset_results.keys():
            across_dataset_results[metric] = np.mean(
                across_dataset_results[metric]
            ).item()

        # Save the average results across all scenes to a json file
        with open(
            os.path.join(
                args.output_dir, f"{benchmark_dataset_name}_avg_across_all_scenes.json"
            ),
            "w",
        ) as f:
            json.dump(across_dataset_results, f, indent=4)

        # log.info the average results across all scenes
        log.info("Average results across all scenes for dataset: ", benchmark_dataset_name)
        for metric in across_dataset_results.keys():
            log.info(f"{metric}: {across_dataset_results[metric]}")

        # Add the average result to the per dataset result dictionary
        per_dataset_results[benchmark_dataset_name] = across_dataset_results

    # Compute the average results across all datasets and add an average entry to the per dataset result dictionary
    average_results = {}
    for metric in per_dataset_results[next(iter(per_dataset_results))].keys():
        metric_values = [
            per_dataset_results[dataset][metric] for dataset in per_dataset_results
        ]
        average_results[metric] = np.mean(metric_values).item()
    per_dataset_results["Average"] = average_results

    # log.info the average results across all datasets
    log.info("Benchmarking Done! ...")
    log.info("Average results across all datasets:")
    for metric in average_results.keys():
        log.info(f"{metric}: {average_results[metric]}")

    # Save the per dataset results to a json file
    with open(os.path.join(args.output_dir, "per_dataset_results.json"), "w") as f:
        json.dump(per_dataset_results, f, indent=4)
