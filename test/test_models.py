import argparse
import torch
from src.models import *

def test_model(device, model_id="dinov3/base", **kwargs):
    model = get_model_from_id(model_id, **kwargs)
    model.to(device)
    model.eval()
    batch_size = 2
    heights, widths = [224, 490, 480], [224, 448, 640] # original, uneven but divisible, uneven and not divisible
    for h, w in zip(heights, widths):
        images = torch.randint(0, 1, (batch_size, 3, h, w)).to(device)
        with torch.no_grad():
            patch_tokens = model(images)
            print(f"{model_id} - Input shape: {images.shape}, Patch tokens shape: {patch_tokens.shape}")
    
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

    model_ids = args.model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for model_id in model_ids:
        print(f"Testing model {model_id}")
        test_model(model_id=model_id, device=device, with_registers=args.with_registers)
