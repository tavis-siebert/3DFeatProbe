import sys
import argparse
from pathlib import Path
from hydra import initialize, compose
from omegaconf import OmegaConf
from src.training.trainers import VGGTTrainer

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="train_vggt",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name=args.config)
        OmegaConf.resolve(cfg)

    trainer = VGGTTrainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()