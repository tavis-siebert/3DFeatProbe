import torch

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

def unproject(
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
        poses (torch.Tensor): (Inverted i.e. cam-2-wolrd) camera pose per view (N, 4, 4)
        depth_maps (torch.Tensor): Depth maps per view (N, H, W)
        metric_depth (bool): If True, depth maps are in mm 
    Returns:
        Per view unprojected points in world coordinates (N, H, W, 3)
    """
    N, H, W = depth_maps.shape
    y = torch.arange(0, H).to(depth_maps.device)
    x = torch.arange(0, W).to(depth_maps.device)
    y, x = torch.meshgrid(y, x, indexing="ij")

    x = x.unsqueeze(0).repeat(N, 1, 1).view(N, H*W)     # (N, H*W)
    y = y.unsqueeze(0).repeat(N, 1, 1).view(N, H*W)     # (N, H*W)

    fx = intrinsics[:, 0, 0].unsqueeze(-1).expand(-1, H*W)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).expand(-1, H*W)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).expand(-1, H*W)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).expand(-1, H*W)

    z = depth_maps.view(N, H*W)       # (N, H*W)
    if metric_depth: z = z / 1000
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)

    unproj_pts = (poses @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)       # (N, H*W, 4)
    unproj_pts = unproj_pts[..., :3] / unproj_pts[..., 3].unsqueeze(-1)   # (N, H*W, 3)
    unproj_pts = unproj_pts.view(N, H, W, 3)

    return unproj_pts