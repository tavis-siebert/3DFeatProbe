
import torch
from typing import Dict

from external.uco3d.uco3d import UCO3DDataset
from utils.camera import unproject

def discrete_coords_new(world_coords: torch.Tensor, voxel_size: float=0.1):
    """
    Discretize world points to voxel grid coordinates
    Code adapted from: https://github.com/Visual-AI/3DRS/blob/11ad004a9d81d7bdb0034cf19bdca457146e8892/llava/model/language_model/.ipynb_checkpoints/llava_qwen-checkpoint.py#L283

    Args:
        world_coords (torch.Tensor): Per view unprojected points in world coordinates (N, H, W, 3)
        voxel_size (float): Size of each voxel
    """
    min_xyz_range = [-15, -15, -5]
    max_xyz_range = [15, 15, 5]

    min_xyz_range = torch.tensor(min_xyz_range).to(world_coords.device)
    max_xyz_range = torch.tensor(max_xyz_range).to(world_coords.device)

    world_coords = torch.maximum(world_coords, min_xyz_range)
    world_coords = torch.minimum(world_coords, max_xyz_range)
    world_coords_discrete = (world_coords - min_xyz_range) / voxel_size
    world_coords_discrete = world_coords_discrete.round()

    return world_coords_discrete.detach()

def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    Code adapted from: https://github.com/Visual-AI/3DRS/blob/11ad004a9d81d7bdb0034cf19bdca457146e8892/llava/model/language_model/.ipynb_checkpoints/llava_qwen-checkpoint.py#L297 
    """
    assert len(arr.shape) == 3
    arr -= arr.min(1, keepdims=True)[0]
    arr_max = arr.max(1, keepdims=True)[0] + 1

    keys = torch.zeros(arr.shape[0], arr.shape[1], dtype=arr.dtype).to(arr.device)

    # Fortran style indexing
    for j in range(arr.shape[2] - 1):
        keys += arr[..., j]
        keys *= arr_max[..., j + 1]
    keys += arr[..., -1]
    return keys

def compute_correspondence_score(
    dataset: UCO3DDataset
) -> Dict[str, float]:
    """
    Compute correspondence score of features

    Args:
        dataset (UCO3DDataset): The dataset containing 3D points and frames.

    Returns:
        a dict with correspondence scores for each sequence in the dataset
    """
    scores = {}
    for sequence_name in dataset.sequence_names:
        views, depth_maps, poses, intrinsics = [], [], [], []
        for frame_idx in dataset.sequence_indices_in_order[sequence_name]:
            frame_data = dataset[frame_idx]
            image = frame_data.image_rgb
            depth_map = frame_data.depth_map
            # check dimensions, convert if needed, and append
            camera = frame_data.camera
            # TODO: convert from NDC, extract intrinsics and pose

        # TODO: Unproject depth maps to 3D points (might need to flip channel)

        # TODO: Run inference on views to get features

