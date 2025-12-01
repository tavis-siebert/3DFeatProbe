from .base import FeatureExtractor
from .dino import DINOv2, DINOv3
from .clip import CLIP
from .mae import MAE
from .mum import MuMVisionTransformer
from .vggt import VGGTFeatureExtractor

__all__ = [
    "DINOv2", "DINOv3", "CLIP", "MAE", 
    "MuMVisionTransformer", "VGGTFeatureExtractor"
    "FeatureExtractor", "get_model_from_id"
]

def get_extractor_from_id(model_id, model_cfg):
    match model_id.lower():
        case "dinov2":
            return DINOv2(**model_cfg)
        case "dinov3":
            return DINOv3(**model_cfg)
        case "clip":
            return CLIP(**model_cfg)
        case "mae":
            return MAE(**model_cfg)
        case "mumvisiontransformer":
            return MuMVisionTransformer(**model_cfg)
        case "vggtfeatureextractor":
            return VGGTFeatureExtractor(**model_cfg)
        case _:
            raise ValueError(f"Model type '{model_id}' is unknown")