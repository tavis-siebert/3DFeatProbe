import torch
from typing import Tuple

def build_intrinsic_matrix(
    fx: float, 
    fy: float, 
    cx: float, 
    cy: float, 
    device: str | torch.device | None=None, 
    dtype: torch.dtype=torch.float32
) -> torch.Tensor:
    """
    Build 3x3 intrinsic matrix
    """
    K = torch.eye(3, dtype=dtype, device=device)
    K[0,0] = float(fx)
    K[1,1] = float(fy)
    K[0,2] = float(cx)
    K[1,2] = float(cy)
    return K

def build_pose_matrix(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Build 3x4 pose matrix from rotation and translation

    Args:
        R (torch.Tensor): rotation matrix (3, 3)
        t (torch.Tensor): translation vector (3, 1)
    """
    pose = torch.zeros(3, 4, dtype=R.dtype, device=R.device)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose

def invert_pose(pose: torch.Tensor) -> torch.Tensor:
    """
    Return inverse of 3x4 or 4x4 pose matrix (supports batched input)

    Args:
        pose (torch.Tensor): the camera pose (..., 3, 4) or (..., 4, 4)
    Returns:
        pose inverse (torch.Tensor): the inverse pose of shape (..., 4, 4)
    """
    # Convert to 4×4 if needed
    if pose.shape[-2:] == (3, 4):
        bottom = torch.tensor([0,0,0,1], dtype=pose.dtype, device=pose.device)
        bottom = bottom.expand(*pose.shape[:-2], 1, 4)
        pose4 = torch.cat([pose, bottom], dim=-2)
    elif pose.shape[-2:] == (4, 4):
        pose4 = pose
    else:
        raise ValueError("Pose must be (..., 3, 4) or (..., 4, 4)")

    R = pose4[..., :3, :3]
    t = pose4[..., :3, 3]

    R_inv = R.transpose(-1, -2)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)

    inv = torch.eye(4, dtype=pose4.dtype, device=pose4.device)
    inv = inv.expand_as(pose4).clone()
    inv[..., :3, :3] = R_inv
    inv[..., :3, 3] = t_inv

    return inv

def depthmap_to_camera_frame(depthmap: torch.Tensor, intrinsics) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert depth image to a pointcloud in camera frame.

    Args:
        depthmap (torch.Tensor): H,W or B,H,W depthmap
        intrinsics: 3,3 or B,3,3 camera intrinsics

    Returns:
        pointmap in camera frame (H,W,3 or B,H,W,3 tensor), and a mask specifying valid pixels.
    """
    # Add batch dimension if not present
    if depthmap.dim() == 2:
        depthmap = depthmap.unsqueeze(0)
        intrinsics = intrinsics.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size, height, width = depthmap.shape
    device = depthmap.device

    # Compute 3D point in camera frame associated with each pixel
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device).float(),
        torch.arange(height, device=device).float(),
        indexing="xy",
    )
    x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1)

    fx = intrinsics[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics[:, 1, 2].view(-1, 1, 1)

    depth_z = depthmap
    xx = (x_grid - cx) * depth_z / fx
    yy = (y_grid - cy) * depth_z / fy
    pts3d_cam = torch.stack((xx, yy, depth_z), dim=-1)

    # Compute mask of valid non-zero depth pixels
    valid_mask = depthmap > 0.0

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        pts3d_cam = pts3d_cam.squeeze(0)
        valid_mask = valid_mask.squeeze(0)

    return pts3d_cam, valid_mask

def depthmap_to_world_coords(
    intrinsics: torch.Tensor, 
    poses: torch.Tensor, 
    depth_maps: torch.Tensor,
    metric_depth: bool=False,
) -> torch.Tensor:
    """
    Unproject deph map to 3D points
    Code adapted from: https://github.com/Visual-AI/3DRS/blob/main/llava/video_utils.py#L38

    Args:
        intrinsics (torch.Tensor): Camera intrinsics per view (N, 3, 3)
        poses (torch.Tensor): Inverted (i.e. cam-2-wolrd) camera pose per view (N, 4, 4)
        depth_maps (torch.Tensor): Depth maps per view (N, H, W)
        metric_depth (bool): If True, depth maps are in mm 
    Returns:
        Per view unprojected points in world coordinates (N, H, W, 3)
    """
    N, H, W = depth_maps.shape
    if metric_depth:
        depth_maps = depth_maps / 1000.0

    pts_cam, _ = depthmap_to_camera_frame(depth_maps, intrinsics)
    pts_cam_flat = pts_cam.view(N, H * W, 3)

    # convert to homogeneous coordinates
    ones = torch.ones((N, H * W, 1), device=pts_cam.device)
    pts_h = torch.cat([pts_cam_flat, ones], dim=-1)  # (N, H*W, 4)

    # transform using pose matrices (cam2world)
    pts_world_h = poses @ pts_h.transpose(1, 2)       # (N, 4, H*W)
    pts_world_h = pts_world_h.transpose(1, 2)         # (N, H*W, 4)

    # convert back to Euclidean coordinates
    pts_world = pts_world_h[..., :3] / pts_world_h[..., 3:].clamp(min=1e-8)
    pts_world = pts_world.view(N, H, W, 3)

    return pts_world