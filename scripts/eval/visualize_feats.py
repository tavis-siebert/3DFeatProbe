import argparse
from hydra import initialize, compose
from omegaconf import OmegaConf

from src.eval.vis_feats import visualize_features

def main():
    parser = argparse.ArgumentParser(description="Visualize features using pca, kmeans, or correspondence")
    parser.add_argument(
        "--config", 
        type=str, 
        default="visualize_feats",
        help="Name of the config file (without .yaml extension)"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name=args.config)
        OmegaConf.resolve(cfg)
        # Allow the config to be editable
        cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    # Run the testing
    visualize_features(cfg)

if __name__ == "__main__":
    main()  # noqa