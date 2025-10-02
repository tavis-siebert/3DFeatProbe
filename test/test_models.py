import argparse
import torch
from src.models import *

def test_model(device, model_name="dinov2", backbone="base", with_registers=False):
    if model_name == 'dinov2':
        model = DINOv2(backbone, with_registers)
    elif model_name == 'dinov3':
        model = DINOv3(backbone)
    elif model_name == 'clip':
        model = CLIP(backbone)
    elif model_name == 'siglip2':
        model = SigLIP2(backbone)
    elif model_name == "mae":
        model = MAE(backbone)
    else:
        raise ValueError(f"Model of type {model_name} is unknown")

    model.to(device)
    model.eval()
    batch_size = 2
    heights, widths = [224, 490, 480], [224, 448, 640] # original, uneven but divisible, uneven and not divisible
    for h, w in zip(heights, widths):
        images = torch.randint(0, 1, (batch_size, 3, h, w)).to(device)
        with torch.no_grad():
            patch_tokens = model(images)
            print(f"{model_name} - Input shape: {images.shape}, Patch tokens shape: {patch_tokens.shape}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        nargs="+",   # accept one or more values
        required=True,
        help="Model name(s) to use, or 'all' to use every available model."
    )
    parser.add_argument(
        "--with_registers",
        action="store_true",
        help="Whether you want to test a model with register tokens. This only matters for specific models like dinov2"
    )
    args = parser.parse_args()

    model_names = args.model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for model_name in model_names:
        name, backbone = model_name.split('/')
        print(f"Testing model {model_name}")
        test_model(device, name, backbone, args.with_registers)
