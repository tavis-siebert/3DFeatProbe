from .vggt import VGGT
from .backbones import get_backbonel_from_id

import torch.nn as nn
from typing import Dict

__all__ = ["get_model_from_model_id"]

def get_model_from_model_id(model_id: str, model_cfg: Dict) -> nn.Module:
    """
    Factory function to return a model instance based on model_id.
    See src/models/README.md for a full list of models and their ids

    Args:
        model_id (str): should be in fromat model_type/...
        model_cfg (Dict): the model config defining parameters and architecture specifics
    """
    model_type = model_id.split('/')[0]
    match model_type.lower():
        case "backbone":
            return get_backbonel_from_id('/'.join(model_id[1:]), model_cfg)
        case "vggt":
            return VGGT(**model_cfg)