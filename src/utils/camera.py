
import torch

def build_intrinsic_matrix(
    fx: float, 
    fy: float, 
    cx: float, 
    cy: float, 
    device: str | torch.device | None=None, 
    dtype: torch.dtype=torch.float32
) -> torch.Tensor:
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
        R (torch.Tensor): rotation matrix
        t (torch.Tensor): translation vector
    """
    pose = torch.zeros(3, 4, dtype=R.dtype, device=R.device)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose

def invert_pose(pose4: torch.Tensor) -> torch.Tensor:
    """
    Return inverse of 4x4 homogeneous pose

    Args:
        pose4 (torch.Tensor): the camera pose (in homogeneous coordinates)
    """
    R = pose4[:3, :3]
    t = pose4[:3, 3]
    R_inv = R.t()
    t_inv = -R_inv @ t
    inv = torch.eye(4, dtype=pose4.dtype, device=pose4.device)
    inv[:3, :3] = R_inv
    inv[:3, 3] = t_inv
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
        intrinsics (torch.Tensor): Camera intrinsics per view (N, 4, 4)
        poses (torch.Tensor): (Inverted i.e. cam-2-wolrd) camera pose per view (N, 4, 4)
        depth_maps (torch.Tensor): Depth maps per view (B, H, W)
        metric_depth (bool): If True, depth maps are in mm 
    Returns:
        Per view unprojected points in world coordinates (N, H, W, 3)
    """
    N, H, W = depth_maps.shape
    y = torch.arange(0, H).to(depth_maps.device)
    x = torch.arange(0, W).to(depth_maps.device)
    y, x = torch.meshgrid(y, x)

    x = x.unsqueeze(0).repeat(N, 1, 1).view(N, H*W)     # (N, H*W)
    y = y.unsqueeze(0).repeat(N, 1, 1).view(N, H*W)     # (N, H*W)

    fx = intrinsics[:, 0, 0].unsqueeze(-1).repeat(1, H*W)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).repeat(1, H*W)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).repeat(1, H*W)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).repeat(1, H*W)

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