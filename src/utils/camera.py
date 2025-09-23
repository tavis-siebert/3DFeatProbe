
import torch

def unproject(intrinsics: torch.Tensor, poses: torch.Tensor, depth_maps: torch.Tensor) -> torch.Tensor:
    """
    Unproject deph map to 3D points
    Code adapted from: https://github.com/Visual-AI/3DRS/blob/main/llava/video_utils.py#L38

    Args:
        intrinsics (torch.Tensor): Camera intrinsics per view (N, 4, 4)
        poses (torch.Tensor): Camera pose per view (N, 4, 4)
        depth_maps (torch.Tensor): Depth maps per view (B, H, W)
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

    z = depth_maps.view(N, H*W) / 1000       # (N, H*W)
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1) 

    unproj_pts = (poses @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)       # (N, H*W, 4)
    unproj_pts = unproj_pts[..., :3] / unproj_pts[..., 3].unsqueeze(-1)   # (N, H*W, 3)
    unproj_pts = unproj_pts.view(N, H, W, 3)

    return unproj_pts