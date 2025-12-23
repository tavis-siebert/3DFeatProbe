import torch
import numpy as np
from mapanything.utils.metrics import (
    calculate_auc_np,
    evaluate_ate,
    l2_distance_of_unit_ray_directions_to_angular_error,
    m_rel_ae,
    se3_to_relative_pose_error,
    thresh_inliers,
)

def compute_pointmaps_abs_rel(gt_info, pr_info, valid_masks, n_views, batch_idx):
    """Mean relative absolute error for pointmaps"""
    scores = []
    for view_idx in range(n_views):
        valid_mask = valid_masks[view_idx][batch_idx].numpy()
        score = m_rel_ae(
            gt=gt_info["pts3d"][view_idx][batch_idx].numpy(),
            pred=pr_info["pts3d"][view_idx][batch_idx].numpy(),
            mask=valid_mask,
        )
        scores.append(score)
    return np.mean(scores)


def compute_pointmaps_inlier_thres_103(gt_info, pr_info, valid_masks, n_views, batch_idx):
    """Inlier ratio at 1.03 threshold for pointmaps"""
    scores = []
    for view_idx in range(n_views):
        valid_mask = valid_masks[view_idx][batch_idx].numpy()
        score = thresh_inliers(
            gt=gt_info["pts3d"][view_idx][batch_idx].numpy(),
            pred=pr_info["pts3d"][view_idx][batch_idx].numpy(),
            mask=valid_mask,
            thresh=1.03,
        )
        scores.append(score)
    return np.mean(scores)


def compute_pose_ate_rmse(gt_info, pr_info, valid_masks, n_views, batch_idx):
    """Absolute Trajectory Error (ATE) RMSE"""
    gt_poses = [gt_info["poses"][i][batch_idx] for i in range(n_views)]
    pr_poses = [pr_info["poses"][i][batch_idx] for i in range(n_views)]
    ate_score = evaluate_ate(gt_traj=gt_poses, est_traj=pr_poses)
    return ate_score.item()

def compute_pose_auc(gt_info, pr_info, valid_masks, n_views, batch_idx, threshold=30):
    gt_poses = [gt_info["poses"][i][batch_idx] for i in range(n_views)]
    pr_poses = [pr_info["poses"][i][batch_idx] for i in range(n_views)]
    gt_poses_stacked = torch.stack(gt_poses)
    pr_poses_stacked = torch.stack(pr_poses)
    
    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
        pred_se3=pr_poses_stacked,
        gt_se3=gt_poses_stacked,
        num_frames=n_views,
    )
    rError = rel_rangle_deg.cpu().numpy()
    tError = rel_tangle_deg.cpu().numpy()
    auc_score, _ = calculate_auc_np(rError, tError, max_threshold=threshold)
    return auc_score * 100.0

def compute_pose_auc_5(gt_info, pr_info, valid_masks, n_views, batch_idx):
    """Pose AUC@5"""
    return compute_pose_auc(gt_info, pr_info, valid_masks, n_views, batch_idx, threshold=5)

def compute_pose_auc_30(gt_info, pr_info, valid_masks, n_views, batch_idx):
    """Pose AUC@30"""
    return compute_pose_auc(gt_info, pr_info, valid_masks, n_views, batch_idx, threshold=30)

def compute_z_depth_abs_rel(gt_info, pr_info, valid_masks, n_views, batch_idx):
    """Mean relative absolute error for z-depth"""
    scores = []
    for view_idx in range(n_views):
        valid_mask = valid_masks[view_idx][batch_idx].numpy()
        score = m_rel_ae(
            gt=gt_info["z_depths"][view_idx][batch_idx].numpy(),
            pred=pr_info["z_depths"][view_idx][batch_idx].numpy(),
            mask=valid_mask,
        )
        scores.append(score)
    return np.mean(scores)


def compute_z_depth_inlier_thres_103(gt_info, pr_info, valid_masks, n_views, batch_idx):
    """Inlier ratio at 1.03 threshold for z-depth"""
    scores = []
    for view_idx in range(n_views):
        valid_mask = valid_masks[view_idx][batch_idx].numpy()
        score = thresh_inliers(
            gt=gt_info["z_depths"][view_idx][batch_idx].numpy(),
            pred=pr_info["z_depths"][view_idx][batch_idx].numpy(),
            mask=valid_mask,
            thresh=1.03,
        )
        scores.append(score)
    return np.mean(scores)


def compute_ray_dirs_err_deg(gt_info, pr_info, valid_masks, n_views, batch_idx):
    """Angular error in ray directions (calibration)"""
    scores = []
    for view_idx in range(n_views):
        ray_dirs_l2 = torch.norm(
            gt_info["ray_directions"][view_idx][batch_idx]
            - pr_info["ray_directions"][view_idx][batch_idx],
            dim=-1,
        )
        ray_err_deg = l2_distance_of_unit_ray_directions_to_angular_error(ray_dirs_l2)
        score = torch.mean(ray_err_deg).cpu().numpy()
        scores.append(score)
    return np.mean(scores)