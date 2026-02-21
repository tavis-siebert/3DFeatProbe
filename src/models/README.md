# Models

All models are instantiated through factory functions in `src/models/__init__.py` using a `model_id` string. The ID follows the pattern `<model_type>/<model_name>`, which is resolved by `get_model_from_model_id()`.

## Model Types

There are two top-level model types:

| `model_type` | Description |
|---|---|
| `feature_extractor` | Frozen VFM backbones for feature probing (multiview correspondence, visualization) |
| `vggt` | Full VGGT model for end-to-end 3D tasks (reconstruction, calibration, training) |

---

## Feature Extractors

Feature extractors are wrappers around pretrained vision foundation models. They all inherit from `FeatureExtractor` (`src/models/feature_extractors/base.py`) and implement a standardized output schema (for ease of use in eval and larger architectures):

```python
# Single-layer output (most models)
{
    "x_norm": torch.Tensor,            # Full last hidden state (CLS + registers + patches)
    "x_norm_clstoken": torch.Tensor,   # CLS token
    "x_norm_patchtokens": torch.Tensor, # Patch embeddings (B, P, D) or (B, H_p, W_p, D)
}

# Multi-layer output (VGGTFeatureExtractor)
{
    "<layer_name>": {
        "x_norm_patchtokens": torch.Tensor,  # Patch embeddings for this layer
    },
    ...
}
```

### Available Feature Extractors

#### DINOv2

| Field | Value |
|---|---|
| **model_id** | `feature_extractor/dinov2` |
| **Class** | `DINOv2` |
| **Source** | HuggingFace (`facebook/dinov2-*`) or `timm` |
| **Backbones** | `small`, `base`, `large`, `giant` |
| **Patch size** | 14 |

**Config options:**
- `backbone` (str): One of `small`, `base`, `large`, `giant`. Default: `"base"`
- `use_timm` (bool): Use `timm` instead of HuggingFace. Default: `false`
- `with_registers` (bool): Use variant with register tokens. Default: `false`
- `checkpoint_path` (str): Path to custom checkpoint. Default: `null` (uses pretrained)
- `preprocess_images` (bool): Apply internal preprocessing. Default: `true`

#### DINOv3

| Field | Value |
|---|---|
| **model_id** | `feature_extractor/dinov3` |
| **Class** | `DINOv3` |
| **Source** | `timm` |
| **Backbones** | `small`, `small_plus`, `base`, `large`, `huge`, `huge_plus`, `7b` |
| **Patch size** | 16 |

**Config options:**
- `backbone` (str): One of `small`, `small_plus`, `base`, `large`, `huge`, `huge_plus`, `7b`. Default: `"base"`
- `checkpoint_path` (str): Path to custom checkpoint. Default: `null`
- `preprocess_images` (bool): Default: `true`

#### CLIP

| Field | Value |
|---|---|
| **model_id** | `feature_extractor/clip` |
| **Class** | `CLIP` |
| **Source** | HuggingFace (`openai/clip-*`) |
| **Backbones** | `vit-large-patch14`, `vit-base-patch16`, `vit-base-patch32` |
| **Patch size** | 14 or 16 or 32 (depends on backbone) |

**Config options:**
- `backbone` (str): One of `vit-large-patch14`, `vit-base-patch16`, `vit-base-patch32`. Default: `"vit-base-patch16"`
- `checkpoint_path` (str): Path to custom checkpoint. Default: `null`
- `preprocess_images` (bool): Default: `true`

#### MAE

| Field | Value |
|---|---|
| **model_id** | `feature_extractor/mae` |
| **Class** | `MAE` |
| **Source** | HuggingFace (`facebook/vit-mae-*`) |
| **Backbones** | `base`, `large`, `huge` |
| **Patch size** | 16 |

**Config options:**
- `backbone` (str): One of `base`, `large`, `huge`. Default: `"base"`
- `checkpoint_path` (str): Path to custom checkpoint. Default: `null`
- `preprocess_images` (bool): Default: `true`

> Note: Masking is disabled (`mask_ratio=0.0`) so all patches are used for feature extraction.

#### MuM

| Field | Value |
|---|---|
| **model_id** | `feature_extractor/mumvisiontransformer` |
| **Class** | `MuMVisionTransformer` |
| **Source** | [davnords/mum](https://github.com/davnords/mum) (cloned into `external/`) |
| **Backbone** | ViT-L/16 (only variant) |
| **Patch size** | 16 |

**Config options:**
- `checkpoint_path` (str): Path to custom checkpoint. Default: `null`
- `preprocess_images` (bool): Default: `true`

#### VGGTFeatureExtractor

| Field | Value |
|---|---|
| **model_id** | `feature_extractor/vggtfeatureextractor` |
| **Class** | `VGGTFeatureExtractor` |
| **Source** | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| **Backbone** | VGGT-1B (DINOv2-Large + Aggregator) |
| **Patch size** | 14 |

This extractor returns **multi-layer** output — one feature dict per aggregator layer (and optionally the patch embedding layer). This is useful for probing how 3D awareness evolves across the depth of the VGGT aggregator.

**Config options:**
- `vggt_config` (Dict): Config dict for the underlying VGGT model. Default: `null` (loads official 1B pretrained)
- `checkpoint_path` (str): Path to custom VGGT checkpoint. Default: `null`
- `layer_types` (List[str]): Which layers to extract. Options:
  - `"patch_embed"` — Final layer of the patch embedding encoder
  - `"frame"` — All local (frame) attention layers in the aggregator
  - `"global"` — All global attention layers in the aggregator
  - `"all"` — All of the above (default)

---

## VGGT (Full Model)

The `VGGT` class (`src/models/vggt.py`) is a modified version of the [original VGGT](https://github.com/facebookresearch/vggt) that adds support for swapping in any `FeatureExtractor` as the patch embedding encoder.

| Field | Value |
|---|---|
| **model_id** | `vggt` |
| **Class** | `VGGT` |

### Architecture

```
Images → [Patch Embed (any FeatureExtractor)] → [Aggregator (local + global attention)] → [Heads]
                                                                                            ├── CameraHead → pose encoding
                                                                                            ├── DPTHead (depth) → depth + confidence
                                                                                            ├── DPTHead (point) → 3D points + confidence
                                                                                            └── TrackHead → point tracks
```

### Config options

- `img_size` (int): Image size to train on. Default: `518`
- `patch_size` (int): Patch size. Default: `14`
- `embed_dim` (int): Embedding dimension. Default: `1024`
- `enable_camera` (bool): Train camera head. Default: `true`
- `enable_depth` (bool): Train depth head. Default: `true`
- `enable_point` (bool): Train pointmap head. Default: `false`
- `enable_track` (bool): Train tracking head. Default: `false`
- `patch_embed_config` (Dict): Config for a custom patch embedding model. If `null`, defaults to DINOv2-Large with registers. Example:
  ```yaml
  patch_embed_config:
    model_id: feature_extractor/dinov2
    model_config:
      backbone: "base"
      use_timm: true
      with_registers: true
      preprocess_images: false  # aggregator handles this
  ```
- `aggregator_config` (Dict): Override aggregator architecture (depth, num_heads, mlp_ratio, etc.)
- `camera_head_config`, `depth_head_config`, `point_head_config`, `track_head_config` (Dict): Override head architectures

### Loading

```python
# From official pretrained checkpoint
model = VGGT.from_pretrained("facebook/VGGT-1B")

# From config (random init or custom backbone)
model = VGGT(**model_config)
```

---

## Adding a New Model

1. Create a new class in `src/models/feature_extractors/` that inherits from `FeatureExtractor`
2. Implement `forward_features()` returning the standardized output schema
3. Set `self.model`, `self.patch_size`, `self.embed_dim`, and `self.img_size` in `__init__`
4. Register the new model in `src/models/feature_extractors/__init__.py` by adding a case to `get_extractor_from_id()`
5. Create a config YAML in `configs/model/` that inherits from `feature_extractor` and sets the `model_id`
