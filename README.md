# 3DFeatProbe

**Analyzing 3D Awareness in Vision Foundation Models**

This project investigates how much 3D geometric understanding is encoded in the intermediate representations of modern Vision Transformers (ViTs). The project provides a unified evaluation framework for probing pretrained vision foundation models (VFMs) on explicit 3D tasks: multiview correspondence multiview reconstruction, monocular depth estimation, pose estimation, and single-view camera calibration. This is done via fine-tuning experiments that swap in different VFM backbones as the patch embedding encoder for [VGGT](https://github.com/facebookresearch/vggt).

### Results
If you're curious about the insights we gained from the project, check out the findings [here (report, slides, images)](https://drive.google.com/drive/folders/1leBdwrTA4PG7H9I7zfmr_7-X2PbpMpZn?usp=sharing).

### Models Supported

All models are instantiated via a `model_id` string (e.g., `feature_extractor/dinov2`). Feature extractors inherit from a common `FeatureExtractor` base class with a standardized output schema (CLS token, patch tokens, full hidden state), making it straightforward to add new models.

| Model | model_id | Patch Size | Source |
|---|---|---|---|
| **DINOv2** (S/B/L/G) | `feature_extractor/dinov2` | 14 | HuggingFace / timm |
| **DINOv3** (S/B/L/H/7B) | `feature_extractor/dinov3` | 16 | timm |
| **CLIP** (ViT-B/16, ViT-L/14) | `feature_extractor/clip` | 14/16/32 | HuggingFace |
| **MAE** (B/L/H) | `feature_extractor/mae` | 16 | HuggingFace |
| **MuM** (ViT-L/16) | `feature_extractor/mumvisiontransformer` | 16 | [davnords/mum](https://github.com/davnords/mum) |
| **VGGT** (feature extractor) | `feature_extractor/vggtfeatureextractor` | 14 | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| **VGGT** (full model) | `vggt` | 14 | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |

For backbone options, config parameters, and how to add new models, see [`src/models/README.md`](src/models/README.md).

### Eval

| Benchmark | Script | Description |
|---|---|---|
| **Multiview Correspondence** | `scripts.eval.multiview_correspondence_benchmark` | Measures cosine similarity of patch features for 3D-corresponding vs. non-corresponding patches across views using voxelized world coordinates. |
| **Dense N-View Reconstruction** | `scripts.eval.dense_n_view_benchmark` | Evaluates pointmap accuracy, depth estimation, camera pose (ATE, AUC@5°), and ray direction error. |
| **Single-View Calibration** | `scripts.eval.calibration_benchmark` | Evaluates predicted camera intrinsics (ray direction angular error) across multiple resolutions. |
| **Feature Visualization** | `scripts.eval.visualize_feats` | PCA and k-means visualization of patch features across models. |

### Datasets

Benchmarks and training use the [Map-Anything](https://github.com/facebookresearch/map-anything) WAI data format with scenes from:

- **ETH3D** — High-quality multi-view indoor/outdoor stereo dataset
- **ScanNet++ V2** — Large-scale indoor RGB-D dataset
- **BlendedMVS** — Blended multi-view stereo dataset
- **MVS-Synth** — Synthetic multi-view stereo dataset
- **UnrealStereo4K** — Synthetic stereo from Unreal Engine

---

## Setup

### 1. Clone External Dependencies

```bash
mkdir -p external
cd external
git clone https://github.com/facebookresearch/vggt.git
git clone https://github.com/facebookresearch/map-anything.git
git clone https://github.com/davnords/mum.git
cd ..
```

### 2. Install This Package

From the repository root:

```bash
pip install -e .
```

This installs all dependencies from `pyproject.toml` (PyTorch 2.8, timm, transformers, hydra, open3d, etc.).

### 3. Install External Packages

```bash
cd external/map-anything && pip install -e . --no-deps && cd ../..
cd external/vggt && pip install -e . --no-deps && cd ../..
pip install uniception --no-deps
```

> **Note on MuM:** The `mum` repository cannot be installed as a package but has an `__init__.py`. It is imported directly via `external.mum.mum`. This is why all scripts must be run using **module notation** (see below).

### 4. Configure datasets

To download and process datasets, follow the instructions provided in the Map-Anything [README](https://github.com/facebookresearch/map-anything/blob/main/train.md). Unfortunately, it can be quite painful to do so as WAI is preprocessing-heavy and a work-in-progress, but the upshot is that any dataset converted to WAI can be used with this code.

### 5. Update configs
The config system is built around [Hydra](https://hydra.cc/). 

**Training and Eval**

Configs for training and eval are at the top level of `configs/` but reference sub-configs in `configs/machine` (paths), `configs/model` (model used), `configs/dataset` (datasets used), and `configs/training` (training args). Generally, the sub-configs supply a "default" for each model, dataset, training setup, etc and the top-level configs allow for customization. I decided on this structure for modularity, but it could definitely use refinement in hindsight. For a detailed description of how the configs work, see [`configs/README.md`](configs/README.md).

**Configure Paths**

Once you've setup the data and directories on your machine, it's important to supply the paths to `configs/machine/default.yaml` (or create a new machine config) to point to your local data and directories:

```yaml
root_data_dir: /path/to/datasets
mapanything_metadata_dir: /path/to/mapanything/metadata
checkpoints_dir: /path/to/checkpoints
results_dir: /path/to/results
```

---

## Usage

> **Important:** Always run scripts from the repository root using module notation (`python -m ...`) so that relative imports resolve correctly.

### Evaluation

```bash
# Multiview correspondence benchmark (probing frozen features)
python -m scripts.eval.multiview_correspondence_benchmark

# Dense N-view reconstruction benchmark
python -m scripts.eval.dense_n_view_benchmark
```

Benchmark configs are in `configs/`. Switch models by changing the corresponding config YAMLs or via command-line overrides. See [`configs/README.md`](configs/README.md) for details.

### Training

Fine-tune VGGT with a custom patch embedding backbone using distributed training:

```bash
torchrun --nproc_per_node=auto scripts/training/train_vggt.py --config train_vggt
```

The training config (`configs/train_vggt.yaml`) controls:
- **Backbone swap** — Set `model.model_config.patch_embed_config` to any supported `FeatureExtractor`
- **Aggregator architecture** — Adjust depth, heads, MLP ratio
- **Frozen modules** — Freeze the patch embedding encoder via `training.optim.frozen_submodules`
- **Multi-task loss** — Camera pose, depth, and pointmap losses with confidence weighting
- **Logging** — Integrated W&B logging with depth maps and 3D point cloud visualizations

---

## Project Structure

```
3dfeat-refs/
├── configs/                    # Hydra YAML configs (see configs/README.md)
│   ├── machine/                #   Machine-specific paths
│   ├── model/                  #   Model configs (dinov2, dinov3, mum, vggt, ...)
│   ├── dataset/                #   Dataset configs + benchmark/train dataset mixes
│   │   ├── resolution/         #     Default resolutions per dataset
│   │   ├── blendedmvs/         #     BlendedMVS WAI template
│   │   ├── eth3d/              #     ETH3D WAI template
│   │   ├── scannetppv2/        #     ScanNet++ V2 WAI template
│   │   ├── mvs_synth/          #     MVS-Synth WAI template
│   │   ├── unrealstereo4k/     #     UnrealStereo4K WAI template
│   │   └── mpsd/               #     MPSD WAI template
│   ├── training/               #   Training sub-configs (optimizer, loss, logging, ...)
│   ├── train_vggt.yaml         #   Top-level: VGGT training
│   ├── multiview_correspondence_benchmark.yaml  # Top-level: correspondence eval
│   ├── dense_n_view_benchmark.yaml              # Top-level: reconstruction eval
│   ├── calibration_benchmark.yaml               # Top-level: calibration eval
│   └── visualize_feats.yaml                     # Top-level: feature visualization
├── external/                   # External dependencies (cloned repos)
│   ├── map-anything/
│   ├── vggt/
│   └── mum/
├── src/                        # Core source code
│   ├── models/                 #   (see src/models/README.md)
│   │   ├── feature_extractors/ #     Unified VFM wrappers (DINOv2, DINOv3, CLIP, MAE, MuM, VGGT)
│   │   ├── processors/         #     Image preprocessing (resize, normalize, pad)
│   │   └── vggt.py             #     Modified VGGT with swappable patch embeddings
│   ├── eval/
│   │   ├── multiview_correspondence.py  # Voxel-based correspondence scoring
│   │   ├── dense_n_view.py              # Full reconstruction metrics (pointmaps, depth, pose)
│   │   ├── metrics.py                   # Metric computation helpers
│   │   └── vis_feats.py                 # PCA / k-means feature visualization
│   ├── training/
│   │   ├── trainers/           #     Base trainer + VGGT trainer with DDP support
│   │   ├── losses/             #     Multi-task loss (camera, depth, pointmap)
│   │   ├── distributed.py      #     DDP utilities
│   │   ├── gradient_clip.py    #     Per-module gradient clipping
│   │   └── optimizer.py        #     Optimizer and scheduler setup
│   ├── datasets/               #     WAI dataset wrappers
│   └── utils/                  #     Camera math, image utils, I/O, logging
├── scripts/
│   ├── eval/                   #   Evaluation entry points
│   │   ├── multiview_correspondence_benchmark.py
│   │   ├── dense_n_view_benchmark.py
│   │   ├── calibration_benchmark.py
│   │   └── visualize_feats.py
│   └── training/               #   Training entry points
│       └── train_vggt.py
├── results/                    #   Saved benchmark results (JSON)
│   ├── multiview_correspondence/
│   └── dense_n_view/
└── pyproject.toml
```

---

## Acknowledgments

This project builds on and integrates the following works:

- [VGGT](https://github.com/facebookresearch/vggt) — Visual Geometry Grounded Transformer (Meta)
- [Map-Anything](https://github.com/facebookresearch/map-anything) — Multi-view 3D reconstruction framework and datasets (Meta)
- [MuM](https://github.com/davnords/mum) — Multi-view masked autoencoder
- [Fit3D](https://github.com/ywyue/FiT3D) - Improving 2D Feature Representations by 3D-Aware Fine-Tuning
- [3DRS](https://github.com/Visual-AI/3DRS) - Basis for multiview correspondence

Please check them out as well.
