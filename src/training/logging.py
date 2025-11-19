# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.distributed as dist
import numpy as np
import wandb
import matplotlib.pyplot as plt
import io
from PIL import Image
from typing import Tuple

class MetricTracker:
    """Running stats for a scalar metric."""
    __slots__ = ("metric_name", "total", "count", "value")

    def __init__(self, metric_name: str) -> None:
        self.metric_name = metric_name
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0
        self.value = 0.0  # most recent value

    def update(self, val: float, n: int = 1) -> None:
        self.value = float(val)
        self.total += val * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)
    
    @property
    def val(self) -> float:
        """Alias for most recent value (for compatibility)"""
        return self.value
    
    def synchronize_between_processes(self) -> None:
        """
        Synchronize the metric values across all distributed processes.
        After calling this, total and count will be summed across all ranks.
        """
        if not dist.is_available() or not dist.is_initialized():
            return
        
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
            
        # Create tensors for reduction
        t = torch.tensor([self.total, self.count], dtype=torch.float64, device='cuda')
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t.tolist()
        self.total = t[0]
        self.count = int(t[1])

    def __str__(self) -> str:
        return f"{self.metric_name}: {self.value:.4f} (avg {self.average:.4f})"


def create_depth_grid_visualization(images, gt_depths, pred_depths, valid_masks=None) -> wandb.Image:
    """
    Create a 3-row grid: RGB images, GT depths, predicted depths.
    
    Args:
        images (torch.Tensor | np.ndarray): rgb images of shape (S, 3, H, W) in [0, 1]
        gt_depths (torch.Tensor | np.ndarray): ground truth depth maps of shape (S, H, W)
        pred_depths: (torch.Tensor | np.ndarray): predicted depth maps of shape (S, H, W)
        valid_masks: (torch.Tensor | np.ndarray, optional): valid mask for depths of shape (S, H, W)
        
    Returns:
        wandb.Image object
    """
    # Convert to numpy
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if isinstance(gt_depths, torch.Tensor):
        gt_depths = gt_depths.detach().cpu().numpy()
    if isinstance(pred_depths, torch.Tensor):
        pred_depths = pred_depths.detach().cpu().numpy()
    if valid_masks is not None and isinstance(valid_masks, torch.Tensor):
        valid_masks = valid_masks.detach().cpu().numpy()
    
    S = images.shape[0]
    
    # Create figure with 3 rows
    fig, axes = plt.subplots(3, S, figsize=(3*S, 9))
    if S == 1:
        axes = axes.reshape(3, 1)
    
    # Compute global depth range for consistent colormap
    if valid_masks is not None:
        valid_gt_depths = gt_depths[valid_masks > 0.5]
        valid_pred_depths = pred_depths[valid_masks > 0.5]
    else:
        valid_gt_depths = gt_depths[gt_depths > 0]
        valid_pred_depths = pred_depths[pred_depths > 0]
    
    if len(valid_gt_depths) > 0 and len(valid_pred_depths) > 0:
        vmin = min(valid_gt_depths.min(), valid_pred_depths.min())
        vmax = max(np.percentile(valid_gt_depths, 95), np.percentile(valid_pred_depths, 95))
    else:
        vmin, vmax = 0, 1
    
    for i in range(S):
        # row 1: rgb
        img = images[i].transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'View {i+1}')
        axes[0, i].axis('off')
        
        # row 2: gt depth
        im_gt = axes[1, i].imshow(gt_depths[i], cmap='turbo', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'GT Depth {i+1}')
        axes[1, i].axis('off')
        if i == S - 1:
            plt.colorbar(im_gt, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        # row 3: pred depth
        im_pred = axes[2, i].imshow(pred_depths[i], cmap='turbo', vmin=vmin, vmax=vmax)
        axes[2, i].set_title(f'Pred Depth {i+1}')
        axes[2, i].axis('off')
        if i == S - 1:
            plt.colorbar(im_pred, ax=axes[2, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # convert to wandb.Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return wandb.Image(Image.open(buf))


def create_point_cloud_visualization(
    gt_world_points, 
    pred_world_points, 
    pred_conf,
    valid_masks, 
    images
) -> Tuple[wandb.Object3D, wandb.Object3D]:
    """
    Create side-by-side 3D point cloud visualizations.
    
    Args:
        gt_world_points: np.ndarray of shape (S, H, W, 3)
        pred_world_points: np.ndarray of shape (S, H, W, 3)
        pred_conf: np.ndarray of shape (S, H, W)
        valid_masks: np.ndarray of shape (S, H, W)
        images: np.ndarray of shape (S, 3, H, W) in [0, 1]
        
    Returns:
        wandb.Object3D for GT and Pred point clouds
    """
    S, H, W, _ = gt_world_points.shape
    
    # Flatten all views
    gt_points_flat = gt_world_points.reshape(-1, 3)
    pred_points_flat = pred_world_points.reshape(-1, 3)
    valid_masks_flat = valid_masks.reshape(-1)
    conf_flat = pred_conf.reshape(-1)
    
    # Get colors from images (S, 3, H, W) -> (S, H, W, 3) -> flat
    colors = images.transpose(0, 2, 3, 1)  # (S, H, W, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)

    # Filter by valid mask and confidence values
    valid_idx = valid_masks_flat > 0.5
    thesh_conf = np.percentile(conf_flat, 50)
    conf_idx = (conf_flat >= thesh_conf) & (conf_flat > 0.1)

    gt_points_valid = gt_points_flat[valid_idx]
    pred_points_valid = pred_points_flat[valid_idx & conf_idx]

    colors_gt = colors_flat[valid_idx]
    colors_pred = colors_flat[valid_idx & conf_idx]

    # Create wandb 3D objects
    # format: [[x, y, z, r, g, b], ...] (S * H * W x 6)
    gt_point_cloud = np.concatenate([gt_points_valid, colors_gt], axis=1)
    pred_point_cloud = np.concatenate([pred_points_valid, colors_pred], axis=1)
    
    gt_point_cloud = wandb.Object3D(gt_point_cloud)
    pred_point_cloud = wandb.Object3D(pred_point_cloud)
    return gt_point_cloud, pred_point_cloud