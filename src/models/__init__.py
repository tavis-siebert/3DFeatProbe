
from .dino import DINOv2, DINOv3
from .clip import CLIP
from .siglip import SigLIP2
from .mae import MAE

__all__ = ["DINOv2", "DINOv3", "CLIP", "SigLIP2", "MAE", "get_model_from_id"]

def get_model_from_id(model_id, **kwargs):
    """Factory function to return a model instance based on model_id.
    See src/models/README.md for a full list of models and their ids

    Args:
        model_id (str): Format "name/backbone", e.g. "dinov2/base".
        kwargs: Additional keyword arguments passed to the model constructor.
    """
    try:
        name, backbone = model_id.split('/')
    except ValueError:
        raise ValueError(f"Invalid model_id '{model_id}', expected format 'name/backbone'")

    match name.lower():
        case "dinov2":
            return DINOv2(backbone, kwargs.get("with_registers", False))
        case "dinov3":
            return DINOv3(backbone)
        case "clip":
            return CLIP(backbone)
        case "siglip2":
            return SigLIP2(backbone)
        case "mae":
            return MAE(backbone)
        case _:
            raise ValueError(f"Model type '{name}' is unknown")