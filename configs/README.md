# Configs

The configuration system uses [Hydra](https://hydra.cc/) with a hierarchical structure. Top-level configs (for training and eval scripts) compose sub-configs from the `machine/`, `model/`, `dataset/`, and `training/` directories via Hydra's `defaults` list. It is designed such that adding new models, datasets, optimizers, etc can be a modular process.

## Directory Structure

```
configs/
├── machine/                  # Machine-specific paths
│   └── default.yaml          #   Template with required path placeholders
├── model/                    # Model definitions
│   ├── default.yaml          #   Base model fields (model_id, data_norm_type, model_config)
│   ├── feature_extractor.yaml #  Base config for all feature extractors
│   ├── dinov2.yaml           #   DINOv2 feature extractor
│   ├── dinov3.yaml           #   DINOv3 feature extractor
│   ├── mum.yaml              #   MuM feature extractor
│   ├── vggt.yaml             #   Full VGGT model (for training / reconstruction)
│   └── vggt_feature.yaml     #   VGGT as a feature extractor (for probing)
├── dataset/                  # Dataset definitions
│   ├── default.yaml          #   Base dataset fields (num_views, num_workers, etc.)
│   ├── resolution/
│   │   └── default.yaml      #   Default train/val resolutions per dataset
│   ├── blendedmvs/
│   │   └── default.yaml      #   BlendedMVS dataset string templates
│   ├── eth3d/
│   │   └── default.yaml      #   ETH3D dataset string templates
│   ├── scannetppv2/
│   │   └── default.yaml      #   ScanNet++ V2 dataset string templates
│   ├── mvs_synth/
│   │   └── default.yaml      #   MVS-Synth dataset string templates
│   ├── unrealstereo4k/
│   │   └── default.yaml      #   UnrealStereo4K dataset string templates
│   ├── mpsd/
│   │   └── default.yaml      #   MPSD dataset string templates
│   ├── multi_dataset_train.yaml        # Multi-dataset training mix
│   ├── benchmark_multiview_correspondence.yaml  # Eval dataset config
│   ├── benchmark_dense_n_view.yaml              # Eval dataset config
│   ├── benchmark_calibration.yaml               # Eval dataset config
│   └── visualize_feats.yaml                     # Visualization dataset config
├── training/                 # Training-specific configs
│   ├── default.yaml          #   Base training fields (epochs, batch size, seed)
│   ├── vggt.yaml             #   VGGT-specific training overrides
│   ├── optim/
│   │   └── default.yaml      #   Optimizer, scheduler, AMP, frozen modules
│   ├── loss/
│   │   └── default.yaml      #   Loss function placeholder
│   ├── logging/
│   │   └── default.yaml      #   W&B config, log frequency, metrics to track
│   ├── checkpoint/
│   │   └── default.yaml      #   Save frequency, resume path, job ID
│   └── distributed/
│       └── default.yaml      #   DDP backend and args
│
│ # ── Top-level configs (entry points) ──
├── train_vggt.yaml                          # Training
├── multiview_correspondence_benchmark.yaml  # Eval: correspondence probing
├── dense_n_view_benchmark.yaml              # Eval: dense reconstruction
├── calibration_benchmark.yaml               # Eval: single-view calibration
└── visualize_feats.yaml                     # Feature visualization
```

## How It Works

### Composition via Defaults

Each top-level config specifies which sub-configs to compose. For example, `multiview_correspondence_benchmark.yaml`:

```yaml
defaults:
  - machine: default    # pulls in configs/machine/default.yaml
  - model: dinov2       # pulls in configs/model/dinov2.yaml
  - dataset: benchmark_multiview_correspondence  # pulls in configs/dataset/benchmark_multiview_correspondence.yaml
  - _self_              # this file's own keys override the defaults
```

The sub-configs themselves can also inherit. For instance, `model/dinov2.yaml` inherits from `model/feature_extractor.yaml`, which inherits from `model/default.yaml`:

```
model/default.yaml          →  model_id: ???, data_norm_type: ???, model_config: ???
  └── model/feature_extractor.yaml  →  data_norm_type: "identity", checkpoint_path: null, preprocess_images: true
        └── model/dinov2.yaml       →  model_id: "feature_extractor/dinov2", backbone: "base", ...
```

### Overriding Values

Top-level configs can override any value from their sub-configs. For example, `train_vggt.yaml` overrides the VGGT model's `embed_dim` and adds a custom `patch_embed_config`:

```yaml
model:
  model_config:
    embed_dim: 768
    patch_embed_config:
      model_id: feature_extractor/dinov2
      model_config:
        backbone: "base"
```

You can also override values from the command line:

```bash
python -m scripts.eval.multiview_correspondence_benchmark model=mum dataset.num_views=2
```

### Variable Interpolation

Hydra's `${...}` syntax is used extensively to avoid duplication. Machine paths flow through dataset configs:

```yaml
# machine/default.yaml
root_data_dir: /path/to/data

# dataset/blendedmvs/default.yaml
ROOT: ${root_data_dir}/blendedmvs_wai
```

---

## Sub-Config Details

### `machine/`

Defines all filesystem paths. **You must create or edit a machine config** to point to your local directories.

| Key | Description |
|---|---|
| `root_data_dir` | Root directory containing all WAI-format datasets |
| `mapanything_metadata_dir` | [Map-Anything dataset metadata](https://huggingface.co/datasets/facebook/map-anything/tree/main/mapanything_dataset_metadata) directory |
| `checkpoints_dir` | Where to store/load model checkpoints |
| `logging_dir` | Where to write training logs |
| `results_dir` | Where to write evaluation results |

### `model/`

Model definitions:
- **`default.yaml`** - Every model config must have `model_id`, `data_norm_type`, and `model_config`. 
- **`feature_extractor.yaml`** — Base config for all feature extractors. Sets `data_norm_type: "identity"` (see below) and provides `checkpoint_path` and `preprocess_images` defaults.
- **Individual model configs** (`dinov2.yaml`, `mum.yaml`, etc.) — Inherit from `feature_extractor.yaml` and set the `model_id` and model-specific parameters. See [`src/models/README.md`](../src/models/README.md) for the full list of models and their config options.
- **`vggt.yaml`** — Full VGGT model config with all head and aggregator options.

The `model.data_norm_type` field might be confusing at first. It is taken from [Map-Anything's configs](https://github.com/facebookresearch/map-anything/tree/main/configs) and specifies how the data should be pre-processed by the dataloader (e.g., normalize, resize). As VGGT and all FeatureExtractors come with built-in preprocessing, we set `model.data_norm_type` to `"identity"` for all models. In other words, this project has no use case for other values of `model.data_norm_type`, and is simply included for compatability with Map-Anything datasets.

```yaml
# model/vggt.yaml
data_norm_type: "identity"

# dataset/blendedmvs/default.yaml → val.dataset_str
data_norm_type='${model.data_norm_type}'
```

### `dataset/`

Each dataset directory contains a `default.yaml` that defines template strings for WAI dataset instantiation. These templates use Hydra interpolation to pull in resolution, normalization, split, and path values.

- **`resolution/default.yaml`** — Default train/val resolutions for each dataset. Override these in your top-level config to change image sizes.
- **`benchmark_*.yaml`** — Pre-defined dataset mixes for each evaluation benchmark. These set `test_dataset` to a `+`-separated string of weighted dataset samples (e.g., `"860 @ BlendedMVSWAI(...) + 600 @ ScanNetPPV2WAI(...)"`).
- **`multi_dataset_train.yaml`** — Balanced multi-dataset training mix.

### `training/`

Training hyperparameters, broken into sub-directories:

- **`optim/default.yaml`** — Optimizer (AdamW), learning rate, weight decay, AMP settings, frozen submodules, gradient clipping
- **`loss/default.yaml`** — Loss function placeholder (currently  set by top-level config as originally more models and losses were planned)
- **`logging/default.yaml`** — W&B project/entity/name, log frequency, which metrics to track
- **`checkpoint/default.yaml`** — Checkpoint save frequency, resume path, job ID
- **`distributed/default.yaml`** — DDP backend and args
- **`vggt.yaml`** — VGGT-specific overrides: cosine LR schedule, per-module gradient clipping, multi-task loss weights, metrics to log
