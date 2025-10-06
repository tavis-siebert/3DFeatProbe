"""
Code to test extracting desired properties from datasets
"""
import argparse
import logging
import os
import torch

from src.eval.multiview_correspondence import compute_correspondence_score
from src.utils.logging import init_logger
from src.models import *

logger = logging.getLogger()

def test_uco3d(model, device):
    from external.uco3d.uco3d import UCO3DDataset, UCO3DFrameDataBuilder

    #TODO: use config or .env, not hardcoded path
    dataset_root = "/cluster/scratch/tsiebert/uco3d/"
    subset_lists_file = os.path.join(
        dataset_root,
        "set_lists", 
        "set_lists_all-categories.sqlite"
        # "set_lists_3categories-debug.sqlite",
    )

    dataset = UCO3DDataset(
        subset_lists_file=subset_lists_file,
        subsets=['train'],
        #TODO: wth do all these parameters control
        frame_data_builder=UCO3DFrameDataBuilder(
            dataset_root=dataset_root,
            apply_alignment=True,
            load_images=True,
            load_depths=True,
            load_masks=True,
            load_depth_masks=True,
            load_gaussian_splats=False,
            gaussian_splats_truncate_background=False,
            load_point_clouds=True,
            load_segmented_point_clouds=False,
            load_sparse_point_clouds=False,
            box_crop=True,
            box_crop_context=0.4,
            load_frames_from_videos=True,
            image_height=800,
            image_width=800,
            undistort_loaded_blobs=True
        )
    )

    model.to(device)
    model.eval()

    scores = compute_correspondence_score(dataset, model, device)
    for seq_name, score_dict in scores.items():
        logger.info(f"Sequence: {seq_name}")
        for key, value in score_dict.items():
            logger.info(f"  {key}: {value}")

    logger.info("===== Summary ====")
    corr_scores = torch.tensor([score_dict["corr"] for score_dict in scores.values()])
    non_corr_scores = torch.tensor([score_dict["non_corr"] for score_dict in scores.values()])
    avg_corr_score, avg_non_corr_score = corr_scores.mean().item(), non_corr_scores.mean().item()
    max_corr_score, min_corr_score = corr_scores.max().item(), corr_scores.min().item()
    logger.info(f"Average Corr Score: {avg_corr_score:.4f}")
    logger.info(f"Average Non-Corr Score: {avg_non_corr_score:.4f}")
    logger.info(f"Min Corr Score: {min_corr_score:.4f}")
    logger.info(f"Max Corr Score: {max_corr_score:.4f}")

if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        nargs="+",   # accept one or more values
        required=True,
        help="Model name(s) to use."
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
        model = get_model_from_id(model_id=model_id, with_registers=args.with_registers)
        logger.info(f"Computing correspondence for {model_id}")
        test_uco3d(model, device)
        logger.info("=======================")