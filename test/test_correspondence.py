"""
Code to test extracting desired properties from datasets
"""
import os
import torch
from src.eval.multiview_correspondence import compute_correspondence_score
from src.models import *

def test_uco3d():
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
            load_point_clouds=False,
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DINOv2('base', with_registers=True)
    model.to(device)
    model.eval()

    print(f"Computing Correspondences on uco3d Debug Split with DINOv2")
    print("=======================")
    scores = compute_correspondence_score(dataset, model, device)
    for seq_name, score_dict in scores.items():
        print(f"Sequence: {seq_name}")
        for key, value in score_dict.items():
            print(f"  {key}: {value}")
    print("===== Summary ====")
    avg_corr_score = sum(score_dict["corr"] for score_dict in scores.values()) / len(scores)
    avg_non_corr_score = sum(score_dict["non_corr"] for score_dict in scores.values()) / len(scores)
    print(f"Average Corr Score: {avg_corr_score:.4f}")
    print(f"Average Non-Corr Score: {avg_non_corr_score:.4f}")

if __name__ == '__main__':
    test_uco3d()