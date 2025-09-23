
import torch
from src.models import *

def test_dinov2_forward(device, backbone="base"):
    model = DINOv2(backbone, with_registers=False)
    model.to(device)
    model.eval()
    batch_size = 2
    heights, widths = [224, 490, 480], [224, 448, 640] # original, uneven but divisible, uneven and not divisible
    for h, w in zip(heights, widths):
        images = torch.randint(0, 1, (batch_size, 3, h, w)).to(device)
        with torch.no_grad():
            patch_tokens = model(images)
            print(f"DINOv2 - Input shape: {images.shape}, Patch tokens shape: {patch_tokens.shape}")
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dinov2_forward(device)