
import torch
from src.models import *

def test_dino_forward(device, model_name="dinov2", backbone="base", with_registers=False):
    if model_name == 'dinov2':
        model = DINOv2(backbone, with_registers)
    elif model_name == 'dinov3':
        model = DINOv3(backbone)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dino_forward(device, model_name="dinov2")
    test_dino_forward(device, model_name="dinov3")