import argparse
from hydra import initialize, compose
from omegaconf import OmegaConf

# single view calibration is just dense n-view with 1 view and ray_err metric
from src.eval.dense_n_view import benchmark

def main():
    parser = argparse.ArgumentParser(description="Benchmark model on single view calibration with config file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="calibration_benchmark",
        help="Name of the config file (without .yaml extension)"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name=args.config)
        OmegaConf.resolve(cfg)
        # Allow the config to be editable
        cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    # Run the testing
    benchmark(cfg)

if __name__ == "__main__":
    main()  # noqa