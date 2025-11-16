import logging
import contextlib
import time
import numpy as np
import wandb
from math import isfinite
from omegaconf import DictConfig
from typing import Dict, Union
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from src.datasets.wai_dataset import build_wai_dataloader
from src.models.vggt import VGGT
from src.utils.camera import invert_pose
from src.training.losses.multitask_loss import MultitaskLoss
from src.training.distributed import all_reduce_mean
from src.training.utils import *
from src.training.logging import *
from .trainer import Trainer

class VGGTTrainer(Trainer):
    """
    VGGT trainer class
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    #-------------------#
    #       Setup       #
    #-------------------#
    # -- Setup datasets
    def _setup_datasets(self):
        # train datasets using WAI format
        logging.info("Building train dataset {:s}".format(self.dataset_cfg.train_dataset))
        self.train_loader = build_wai_dataloader(
            dataset=self.dataset_cfg.train_dataset,
            num_workers=self.dataset_cfg.num_workers,
            test=False,
            max_num_of_imgs_per_gpu=self.train_cfg.max_num_of_imgs_per_gpu
        )

        # val datasets using WAI format
        logging.info("Building validation dataset {:s}".format(self.dataset_cfg.test_dataset))
        val_batch_size = 2 * (
            self.train_cfg.max_num_of_imgs_per_gpu // self.dataset_cfg.num_views
        )
        self.val_loader = {
            dataset.split("(")[0]: build_wai_dataloader(
                dataset=dataset,
                num_workers=self.dataset_cfg.num_workers,
                test=True,
                batch_size=val_batch_size,
            )
            for dataset in self.dataset_cfg.test_dataset.split("+")
            if "(" in dataset
        }
    
    # -- Setup model
    def _setup_model(self):
        # initialize VGGT from official checkpoint or model config
        logging.info("Initializing VGGT")
        if self.model_cfg.load_pretrained:
            self.model = VGGT.from_pretrained()
        else:
            model_args = self.model_cfg.model_config
            # check img_size = train_resolution
            train_h, train_w = eval(self.train_cfg.resolution.train)
            if train_h != train_w:
                raise ValueError(f"VGGT must be trained with square images. Got {(train_h, train_w)}")
            if train_h != model_args.img_size:
                logging.warning(f"Got train image size {train_h} and config image size {model_args.img_size}. Defaulting to training size")
                model_args.img_size = train_h

            self.model = VGGT(**model_args)
        logging.info(f"Initialized Model: {str(self.model)}")

        # swap out patch embed
        patch_embed_ckpt_path = self.model_cfg.pretrained_patch_embed_path
        if patch_embed_ckpt_path:
            logging.info(f"Switching patch embed to '{patch_embed_ckpt_path}'")
            self._load_specific_patch_embed(patch_embed_ckpt_path)

        # move to device
        self.model.to(self.device)
    
    def _load_specific_patch_embed(self, ckpt_path: str) -> Dict:
        """
        This is a VERY BRITTLE function for our experiments.
        Hopefully it is clear how the function works and how to structure the config.
        Specifically, ensure to set `pretrained_patch_embed_path` and either rename the checkpoint or change this function
        to accomodate your checkpoint
        
        Args:
            ckpt_path (str): the torch hub or local path to the dinov2 weights that will be used to initialize VGGT's patch_embed
        """
        ckpt_config = {}
        if "fit3d" in ckpt_path.lower():
            patch_embed = torch.hub.load("ywyue/FiT3D", "dinov2_base_fine")
            state_dict = patch_embed.state_dict()
            ckpt_config["checkpoint_state_dict"] = state_dict
            ckpt_config["original_image_size"] = 518
        elif "dinov2" in ckpt_path.lower():
            # assumes a variant of timm DINOv2
            patch_embed = torch.load(ckpt_path, map_location='cpu')
            state_dict = {
                k.replace("model.", ""): v for k, v in patch_embed.state_dict().items()
            }
            ckpt_config["checkpoint_state_dict"] = state_dict
            ckpt_config["original_image_size"] = 518
        else:
            raise ValueError(f"Failed to match '{ckpt_path}' to one of our state dicts")
        
        self.model.load_pretrained_patch_embed(ckpt_config)

        del patch_embed

    # -- Setup loss / criterion
    def _setup_loss(self):
        self.loss = MultitaskLoss(**self.loss_cfg.loss_config)

    #-------------------#
    #   Functionality   #
    #-------------------#

    def _convert_mapa_batch_to_vggt(self, views: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Convert map-anything's list-of-view-dicts format to VGGT's batched format.
        
        Args:
            views: List of view dictionaries from map-anything dataloader.
            
        Returns:
            expected batch for VGGT input and loss
        """ 
        def __convert_from_numpy(arr: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
            if isinstance(arr, np.ndarray):
                return torch.from_numpy(arr)
            return arr

        image_batch, depth_batch, valid_mask_batch = [], [], []
        intrinsics_batch, extrinsics_batch = [], []
        world_pts_batch, cam_pts_batch = [], []

        for view in views:
            # rgb image
            image = view['img'] # [B, 3, H, W]
            image_batch.append(__convert_from_numpy(image))

            # depth map
            depthmap = view['depthmap']  # [B, H, W]
            depth_batch.append(__convert_from_numpy(depthmap))

            # mask
            valid_mask = view['valid_mask']  # [B, H, W]
            valid_mask_batch.append(__convert_from_numpy(valid_mask))

            # camera intrinsics
            intrinsics = view['camera_intrinsics']  # [B, 3, 3]
            intrinsics_batch.append(__convert_from_numpy(intrinsics))

            # camera pose (ggt expects world2cam)
            cam2world = __convert_from_numpy(view['camera_pose']) # [B, 4, 4]
            world2cam = invert_pose(cam2world)
            pose = world2cam[:, :3, :]  # [B, 3, 4]
            extrinsics_batch.append(pose)

            # point maps
            pts3d = view['pts3d']  # [B, H, W, 3]
            pts3d_cam = view["pts3d_cam"]
            world_pts_batch.append(__convert_from_numpy(pts3d))
            cam_pts_batch.append(__convert_from_numpy(pts3d_cam))

        # stack and arrange as (B, num_views, ...)
        image_batch = torch.stack(image_batch, dim=1)
        depth_batch = torch.stack(depth_batch, dim=1)
        valid_mask_batch = torch.stack(valid_mask_batch, dim=1)
        intrinsics_batch = torch.stack(intrinsics_batch, dim=1)
        extrinsics_batch = torch.stack(extrinsics_batch, dim=1)
        world_pts_batch = torch.stack(world_pts_batch, dim=1)
        cam_pts_batch = torch.stack(cam_pts_batch, dim=1)
        
        return {
            "images": image_batch,
            "extrinsics": extrinsics_batch,
            "intrinsics": intrinsics_batch,
            "depths": depth_batch,
            "world_points": world_pts_batch,
            "cam_points": cam_pts_batch,
            "point_masks": valid_mask_batch,
        }

    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process batch with normalization"""
        # Normalize camera extrinsics and points
        normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
            )
        
        batch["extrinsics"] = normalized_extrinsics
        batch["cam_points"] = normalized_cam_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths
        
        return batch

    def train(self):
        while self.epoch < self.max_epochs:
            set_seeds(self.seed + self.epoch * 100, self.rank)
            
            self.train_epoch()

            if (self.epoch + 1) % self.save_freq == 0 and self.rank == 0:
                ckpt_name = self._create_ckpt_name(f"checkpoint-last_vggt")
                self.save_checkpoint(ckpt_name)

            if (self.epoch + 1) % self.eval_freq == 0:
                self.val_epoch()
            
            self.epoch += 1

    def train_epoch(self):
        self.model.train()
        
        # Setup metrics tracking
        batch_time = MetricTracker("Batch Time")
        data_time = MetricTracker("Data Time")
        mem = MetricTracker("Mem (GB)")
        
        loss_names = [f"Loss/train_{name}" for name in self.train_metrics_to_log]
        loss_trackers = {name: MetricTracker(name) for name in loss_names}
        
        if self.optim_cfg.gradient_clip:
            for config in self.gradient_clipper.configs:
                param_names = ",".join(config['module_names'])
                loss_trackers[f"Grad/{param_names}"] = MetricTracker(f"Grad/{param_names}")
        
        # Setup gradient clipping
        if self.optim_cfg.gradient_clip:
            self.gradient_clipper.setup_clipping(self.model)
        
        # Training loop
        trackers_to_display = [batch_time, data_time, mem, *loss_trackers.values()]
        batch_start_time = time.time()

        max_iters = len(self.train_loader) if self.max_train_iters is None else self.max_train_iters

        for data_iter, batch in enumerate(self.train_loader):
            # debugging
            if data_iter > max_iters:
                break

            # Measure data loading time
            data_time.update(time.time() - batch_start_time)
            
            # Process batch
            batch = self._convert_mapa_batch_to_vggt(batch)
            batch = self._process_batch(batch)
            batch = move_data_to_device(batch, self.device, non_blocking=True)
            
            # Gradient accumulation chunking
            chunked_batches = chunk_batch_for_accum_steps(batch, self.accum_steps)
            
            # Run forward/backward (with gradient accumulation if enabled)
            self._run_steps_on_batch_chunks(chunked_batches, "train", loss_trackers)
            
            # Gradient clipping
            if self.optim_cfg.gradient_clip:
                self.scaler.unscale_(self.optims.optimizer)
                grad_norm_dict = self.gradient_clipper(model=self.model)
                for key, grad_norm in grad_norm_dict.items():
                    loss_trackers[f"Grad/{key}"].update(grad_norm)
            
            # Optimizer step
            if self.optim_cfg.amp.enabled:
                self.scaler.step(self.optims.optimizer)
                self.scaler.update()
            else:
                self.optims.optimizer.step()
            
            # Scheduler step
            exact_epoch = self.epoch + float(data_iter) / len(self.train_loader)
            self.where = float(exact_epoch) / self.max_epochs
            if self.where < 1.0:
                self.optims.step_schedulers(self.where)
            
            # Track batch training time
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # Memory tracking
            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            # Display metrics for step
            if (data_iter + 1) % self.log_freq == 0:
                # Reduce losses across ranks for logging
                reduced_losses = {}
                for name, tracker in loss_trackers.items():
                    # Use all_reduce_mean to get global average
                    reduced_loss = all_reduce_mean(tracker.val)
                    reduced_losses[name] = reduced_loss

                # Display progress (with reduced losses)
                if self.rank == 0:
                    metrics_dict = {
                        "Batch Time": batch_time.val,
                        "Data Time": data_time.val,
                        "Mem (GB)": mem.val,
                        **reduced_losses
                    }
                    progress_str = f"Train Epoch {self.epoch + 1} [{data_iter + 1}/{len(self.train_loader)}]"
                    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics_dict.items())
                    logging.info("%s %s", progress_str, metrics_str)

                    # Log reduced losses to wandb
                    wandb.log(reduced_losses, step=self.steps['train'] - 1)

        # Display final metrics
        for tracker in trackers_to_display:
            tracker.synchronize_between_processes()

        if self.rank == 0:
            progress_str = f"Train Epoch {self.epoch + 1} - AVG"
            metrics_dict = {tracker.metric_name: tracker.average for tracker in trackers_to_display}
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics_dict.items())
            logging.info("%s %s", progress_str, metrics_str)


    @torch.no_grad()
    def val_epoch(self):
        if self.val_loader is None:
            return
        
        self.model.eval()

        # Track metrics
        batch_time = MetricTracker("Batch Time")
        data_time = MetricTracker("Data Time")
        mem = MetricTracker("Mem (GB)")
        loss_names = [f"Loss/val_{name}" for name in self.val_metrics_to_log]
        loss_trackers = {name: MetricTracker(name) for name in loss_names}

        amp_type = torch.bfloat16 if self.optim_cfg.amp.amp_dtype == "bfloat16" else torch.float16

        # Validate on each validation dataset (separated unlike training)
        for dataset_name, val_loader in self.val_loader.items():
            # debugging
            max_iters = len(val_loader) if self.max_val_iters is None else self.max_val_iters

            # track losses per dataset
            for loss_tracker in loss_trackers.values():
                loss_tracker.reset()
            
            batch_start_time = time.time()
            
            for data_iter, batch in enumerate(val_loader):
                # debugging
                if data_iter > max_iters:
                    break

                # Measure data loading time
                data_time.update(time.time() - batch_start_time)
                
                # Process batch
                batch = self._convert_mapa_batch_to_vggt(batch)  # to vggt format
                batch = self._process_batch(batch)  # normalize
                batch = move_data_to_device(batch, self.device, non_blocking=True)  # to device
                
                # Forward pass
                with torch.no_grad():
                    with torch.autocast(
                        device=self.device.type,
                        enabled=self.optim_cfg.amp.enabled,
                        dtype=amp_type,
                    ):
                        _, preds = self._step(batch, "val", loss_trackers)
                
                # Track batch inference time
                batch_time.update(time.time() - batch_start_time)
                batch_start_time = time.time()

                # Memory tracking
                if torch.cuda.is_available():
                    mem.update(torch.cuda.max_memory_allocated() // 1e9)
            
            # Display losses across dataset
            for tracker in loss_trackers.values():
                tracker.synchronize_between_processes()

            if self.rank == 0:
                progress_str = f"Val Epoch {self.epoch + 1} ({dataset_name}) - AVG"
                metrics_dict = {tracker.metric_name: tracker.average for tracker in loss_trackers.values()}
                metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics_dict.items())
                logging.info("%s %s", progress_str, metrics_str)

                # Log to wandb
                val_metrics = {f"{name}_{dataset_name}": tracker.average for name, tracker in loss_trackers.items()}

                # only logging visuals for last item of last batch of main process
                images = batch["images"][-1]  # [S, 3, H, W]
                pred_depth = preds["depth"][-1].squeeze()  # [S, H, W]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"][-1], images.shape[-2:])

                if "world_points" in preds:
                    pred_world_points = preds["world_points"][-1].cpu().numpy()  # [S, H, W, 3]
                    pred_conf = preds["world_points_conf"][-1].cpu().numpy() # [S, H, W]
                else:
                    pred_world_points = unproject_depth_map_to_point_map(pred_depth, extrinsic, intrinsic)
                    pred_conf = preds["depth_conf"][-1].cpu().numpy()
                
                gt_ptc, pred_ptc = create_point_cloud_visualization(
                    gt_world_points=batch["world_points"][-1].cpu().numpy(),
                    pred_world_points=pred_world_points,
                    pred_conf=pred_conf,
                    valid_masks=batch["point_masks"][-1].cpu().numpy(),
                    images=images.cpu().numpy(),
                )
                
                depth_visuals = create_depth_grid_visualization(
                    images=images,
                    gt_depths=batch["depths"][-1],
                    pred_depths=pred_depth,
                    valid_masks=batch["point_masks"][-1]
                )
                
                #TODO: how much nesting does wandb allow?
                log_dict = {
                    **val_metrics, 
                    f"Depths/{dataset_name}": depth_visuals, 
                    f"PointClouds/{dataset_name}_gt": gt_ptc,
                    f"PointClouds/{dataset_name}_pred": pred_ptc,
                }
                wandb.log(log_dict, step=self.epoch)

        # Display efficiency across validation
        trackers_to_display = [batch_time, data_time, mem]
        for tracker in trackers_to_display:
            tracker.synchronize_between_processes()

        if self.rank == 0:
            progress_str = f"Val Epoch {self.epoch + 1} - AVG"
            metrics_dict = {tracker.metric_name: tracker.average for tracker in trackers_to_display}
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics_dict.items())
            logging.info("%s %s", progress_str, metrics_str)


    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Dict],
        phase: str,
        loss_trackers: Dict[str, MetricTracker],
    ):
        """Run forward/backward on batch chunks for gradient accumulation"""
        self.optims.zero_grad(set_to_none=True)
        
        accum_steps = len(chunked_batches)
        amp_type = torch.bfloat16 if self.optim_cfg.amp.amp_dtype == "bfloat16" else torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            # DDP no_sync for gradient accumulation
            ddp_context = (
                self.model.no_sync() if (self.distributed and i < accum_steps - 1) 
                else contextlib.nullcontext()
            )
            
            with ddp_context:
                with torch.autocast(
                    device=self.device.type,
                    enabled=self.optim_cfg.amp.enabled,
                    dtype=amp_type,
                ):
                    # Forward pass
                    loss_dict, _ = self._step(chunked_batch, phase, loss_trackers)
                
                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]
                
                # Check for NaN/Inf
                if not isfinite(loss.item()):
                    logging.error(f"Loss is {loss.item()}, stopping training")
                    return
                
                # Scale loss for accumulation
                loss /= accum_steps
                
                # Backward pass
                if self.optim_cfg.amp.enabled:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                loss_trackers[loss_key].update(loss.item() * accum_steps, batch_size)


    def _step(self, batch: Dict, phase: str, loss_trackers: Dict[str, MetricTracker]):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
            A dictionary containing the raw predictions
        """
        # Forward pass
        y_hat = self.model(images=batch["images"])
        
        # Loss computation
        loss_dict = self.loss(y_hat, batch)
        
        # Update loss trackers
        batch_size = batch["images"].shape[0]
        metrics_to_log = self.train_metrics_to_log if phase == "train" else self.val_metrics_to_log
        for key in metrics_to_log:
            if key in loss_dict:
                value = loss_dict[key].item() if torch.is_tensor(loss_dict[key]) else loss_dict[key]
                loss_key = f"Loss/{phase}_{key}"
                if loss_key in loss_trackers:
                    loss_trackers[loss_key].update(value, batch_size)
        
        self.steps[phase] += 1
        return loss_dict, y_hat
    
    #-------------------------#
    #  Logging/Checkpointing  #
    #-------------------------#
    def _create_ckpt_name(self, prefix: str):
        patch_embed_path = self.model_cfg.pretrained_patch_embed_path
        if patch_embed_path:
            model_id = patch_embed_path.split("/")[-1].replace(".pt", '')
            prefix += f"-{model_id}"
        return prefix