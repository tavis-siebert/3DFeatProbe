import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from sklearn.decomposition import PCA
from torch_kmeans import KMeans, CosineSimilarity

from src.models import get_model_from_model_id
from src.training.utils import convert_mapa_batch_to_vggt, move_data_to_device
from src.datasets import build_wai_dataloader

# Logging
import logging
from src.utils.logging import direct_logger_to_stdout
log = logging.getLogger(__name__)
direct_logger_to_stdout(log)

# Kmeans
def vis_kmeans(feats: torch.tensor, n_clusters: int=20) -> List[Image.Image]:
    """
    Compute kmeans label map for a feature map
    Args:
        feats (torch.Tensor): featuremap of shape B,H,W,C or 1,B,H,W,C
        n_clusters (int): number of clusters to visualize
    Returns:
        label_maps (List[Image.Image]): list of label map images for each image in the batch dimension
    """
    # flatten the spatial dim
    if feats.ndim == 5:
        if feats.shape[0] != 1:
            raise RuntimeError("Batch dimension for input of size 5 must be 1")
        feats = feats.squeeze(0)
    B, H, W, D = feats.shape
    feats = feats.reshape(1, -1, D)    # multiview-consistent cluster

    kmeans_engine = KMeans(n_clusters=n_clusters, distance=CosineSimilarity)
    kmeans_engine.fit(feats)
    labels = kmeans_engine.predict(feats)
    labels = labels.reshape(B, H, W).float().cpu().numpy()

    # any way to make the color maps consistent?
    cmap = plt.get_cmap("tab20")
    label_maps = []
    for label in labels:
        label_map = cmap(label / n_clusters)[..., :3]
        label_map = Image.fromarray(np.uint8(label_map * 255))
        label_maps.append(label_map)

    return label_maps

    
def vis_pca(feats: torch.tensor, resize_size: Tuple[int, int]=None) -> List[Image.Image]:
    """
    Compute the PCA visualization for a featuremap
    Args:
        feats (torch.Tensor): featuremap of shape B,H,W,C or 1,B,H,W,C
        resize_size (Tuple[int]): the size of the image to resize the map to
    Returns:
        pca_maps (List[Image.Image]): list of pca images for each image in the batch dimension
    """
    # flatten the spatial dim
    if feats.ndim == 5:
        if feats.shape[0] != 1:
            raise RuntimeError("Batch dimension for input of size 5 must be 1")
        feats = feats.squeeze(0)
    B, H, W, D = feats.shape

    pca = PCA(n_components=3)
    pca_maps = []
    for feat in feats:
        proj_feat = feat.reshape(-1, D).detach().cpu().numpy()
        pca.fit(proj_feat)
        pca_features = pca.transform(proj_feat)
        pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
        pca_features = pca_features * 255

        pca_img = Image.fromarray(pca_features.reshape(H, W, 3).astype(np.uint8))
        if resize_size is not None:
            pca_img = pca_img.resize(resize_size, resample=Image.BILINEAR)
        pca_maps.append(pca_img)

    return pca_maps

# Save the plots
def plot_visuals(
    mode: str,
    images: List[Image.Image],
    feat_vis: Dict,
    title: str="",
    figsize_per_cell=(3, 3),
):
    """
    Creates and returns a (N+1) x M grid:
      - Row 0: input images (multiview)
      - Rows 1..N: one row per model, same views

    Args:
        mode (str): the visualization mode (e.g., "pca", "kmeans")
        images (List[Image.Image]): list of M input images
        feat_vis (Dict): dict mapping model/layer_name -> {mode: List[Image.Image]}
    """
    model_names = list(feat_vis.keys())
    M = len(images)
    N = len(model_names)

    fig, axes = plt.subplots(
        N + 1, M,
        figsize=(figsize_per_cell[0] * M, figsize_per_cell[1] * (N + 1))
    )

    # Ensure axes is always 2D
    if (N + 1) == 1:
        axes = axes[None, :]
    if M == 1:
        axes = axes[:, None]

    # First row: images
    for j, img in enumerate(images):
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"View {j}", fontsize=12)
        axes[0, j].axis("off")

    # Add rows
    for i, model_name in enumerate(model_names, start=1):
        for j, img in enumerate(feat_vis[model_name][mode]):
            axes[i, j].imshow(img)
            if j == 0:
                axes[i, j].set_ylabel(model_name, fontsize=12)
            axes[i, j].axis("off")

    # Add title
    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    # Add row titles
    for i, row_name in enumerate(model_names):
        left_ax = axes[i + 1, 0]    # skip first row
        right_ax = axes[i + 1, -1]

        bbox_l = left_ax.get_position()
        bbox_r = right_ax.get_position()

        x_center = 0.5 * (bbox_l.x0 + bbox_r.x1)
        y = bbox_l.y1 + 0.01   # small fixed offset

        fig.text(
            x_center,
            y,
            row_name,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.close(fig)
    return fig


# Visualize features
@torch.no_grad()
def visualize_features(args):
    # Setup output dir (where visuals are saved)
    log.info("Output Directory: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    # Run visualization
    for dataset_name, data_loader in data_loaders.items():
        dataset_name = dataset_name.replace(' ','')
        log.info(f"Visualizing dataset: {dataset_name}")

        results = {}
        # Init list of metrics for each scene
        unique_scenes = set()
        for scene in data_loader.dataset.dataset.scenes:
            results[scene] = {
                "images": [],
                "features": {}
            }
            unique_scenes.add(scene)
        num_unique_scenes = len(unique_scenes)

        for model_cfg in args.model_configs:
            # Load model
            assert "feature_extractor" in model_cfg.model_id
            model = get_model_from_model_id(
                model_cfg.model_id, model_cfg.model_config
            )
            model.to(device)
            model.eval()

            model_name = model_cfg.model_config.checkpoint_path.split('/')[-1].split('.')[0] \
                if model_cfg.model_config.checkpoint_path else model_cfg.model_id.split('/')[1]

            # Reset processed scenes
            data_loader.dataset.set_epoch(0)
            processed_scenes = set()
            max_scenes = args.max_scenes or num_unique_scenes

            for batch in data_loader:
                # Convert batch to B, S, C, H, W and move to device
                model_ready_batch = convert_mapa_batch_to_vggt(batch)
                model_ready_batch = move_data_to_device(model_ready_batch, device)

                # Get features
                with torch.autocast(device, enabled=args.amp.enabled, dtype=amp_dtype):
                    feats_out = model.forward_features(
                        model_ready_batch["images"],
                        unflatten_patches=True
                    )
                
                # Aggregate layer-wise features
                B, S, _, H, W = model_ready_batch["images"].shape
                output_schema = model.validate_output_schema(feats_out)
                if output_schema == "single":
                    patchtokens = feats_out["x_norm_patchtokens"]
                    layerwise_feats = {"final": patchtokens.reshape(B, S, *patchtokens.shape[1:])}
                elif output_schema == "multi":
                    layerwise_feats = {}
                    for layer_name, layer_out in feats_out.items():
                        patchtokens = layer_out["x_norm_patchtokens"]
                        layerwise_feats[layer_name] = patchtokens.reshape(B, S, *patchtokens.shape[1:])

                # Loop over each multiview set
                for batch_idx in range(B):
                    if len(processed_scenes) >= max_scenes:
                        break
                    
                    # Visualize scene
                    scene = batch[0]["label"][batch_idx]
                    if scene not in processed_scenes:
                        imgs = [
                            Image.fromarray(
                                (model_ready_batch["images"][batch_idx, v]
                                .permute(1, 2, 0)   # -> H, W, 3
                                .cpu()
                                .numpy() * 255).astype(np.uint8)
                            ).resize(
                                (W // 2, H // 2), Image.BILINEAR
                            ) for v in range(S)
                        ]
                        results[scene]["images"] = imgs

                        for layer_name, feats in layerwise_feats.items():
                            scene_feat_layer = feats[batch_idx]  # S, Hp, Wp, C
                            for mode in args.vis_modes:
                                mode = mode.lower()
                                if mode == "pca":
                                    vis = vis_pca(scene_feat_layer)
                                elif mode == "kmeans":
                                    vis = vis_kmeans(scene_feat_layer, args.n_clusters)
                                else:
                                    continue
                                results[scene]["features"].setdefault(f"{model_name}_{layer_name}", {})[mode] = vis
                        
                        processed_scenes.add(scene)
            
            # Remove model from GPU
            del model
            torch.cuda.empty_cache()
        
        # Plot
        for scene_id, scene_data in results.items():
            images, vis = scene_data["images"], scene_data["features"]
            if not images:
                continue
            
            for mode in args.vis_modes:
                fig = plot_visuals(
                    mode, images, vis, f"{dataset_name}/{scene_id}/{mode}"
                )
                save_dir = Path(args.output_dir) / dataset_name / mode
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{scene_id}_views-{args.dataset.num_views}.png"
                fig.savefig(save_path, dpi=200)
        
        results.clear()
        del data_loader
        torch.cuda.empty_cache()