import logging
import contextlib
import time
import wandb
from math import isfinite, isnan
from omegaconf import DictConfig
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from src.datasets.wai_dataset import build_wai_dataloader
from src.models.vggt import VGGT
from src.training.losses.multitask_loss import MultitaskLoss
from src.training.distributed import all_reduce_mean
from src.training.utils import *
from src.training.logging import *
from src.eval.dense_n_view import compute_results_for_batcb
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
        val_batch_size = self.train_cfg.max_num_of_imgs_per_gpu // self.dataset_cfg.num_views
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
            self.model = VGGT.from_pretrained("facebook/VGGT-1B")
            logging.info("Initialized Model from `facebook/VGGT-1B'")
        else:
            model_args = self.model_cfg.model_config
            # check img_size = train_resolution
            train_h, train_w = eval(self.dataset_cfg.resolution.train)
            if train_h != train_w:
                raise ValueError(f"VGGT must be trained with square images. Got {(train_h, train_w)}")
            if train_h != model_args.img_size:
                logging.warning(f"Got train image size {train_h} and config image size {model_args.img_size}. Defaulting to training size")
                model_args.img_size = train_h
            self.model = VGGT(**model_args)
            logging.info(f"Initialized Model: {str(self.model)}")

        # move to device
        self.model.to(self.device)
    
    # -- Setup loss / criterion
    def _setup_loss(self):
        self.loss = MultitaskLoss(**self.loss_cfg.loss_config)

    #-------------------#
    #   Functionality   #
    #-------------------#
    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Convert from MapA/wai format
        batch = convert_mapa_batch_to_vggt(batch)

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

            torch.cuda.reset_peak_memory_stats()

            if (self.epoch + 1) % self.eval_freq == 0:
                self.val_epoch()
            
            self.epoch += 1

            # ensures ckpt picks up on next epoch and correct global steps
            if self.epoch % self.save_freq == 0 and self.rank == 0:
                ckpt_name = self._create_ckpt_name(f"checkpoint-{self.job_id}_last_vggt")
                self.save_checkpoint(ckpt_name)
                ckpt_name = self._create_ckpt_name(f"checkpoint-{self.job_id}_epoch-{self.epoch}_vggt")
                self.save_checkpoint(ckpt_name)

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

        # Set epoch for wai dataset samplers
        if hasattr(self.train_loader, "dataset") and hasattr(self.train_loader.dataset, "set_epoch"):
            self.train_loader.dataset.set_epoch(self.epoch)
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.epoch)
        if hasattr(self.train_loader, "batch_sampler") and hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(self.epoch)
        
        # Training loop
        trackers_to_display = [batch_time, data_time, mem, *loss_trackers.values()]
        batch_start_time = time.time()

        max_iters = len(self.train_loader) if self.max_train_iters is None else self.max_train_iters

        for data_iter, batch in enumerate(self.train_loader):
            # debugging
            if data_iter >= max_iters:
                break

            # Measure data loading time
            data_time.update(time.time() - batch_start_time)
            
            # Process batch
            batch = self._process_batch(batch)
            batch = move_data_to_device(batch, self.device, non_blocking=True)
            
            # Gradient accumulation chunking
            chunked_batches = chunk_batch_for_accum_steps(batch, self.accum_steps)
            
            # Run forward/backward (with gradient accumulation if enabled)
            self._run_steps_on_batch_chunks(chunked_batches, "train", loss_trackers)
            self.steps["train"] += 1
            
            # Gradient clipping
            if self.optim_cfg.gradient_clip:
                self.scaler.unscale_(self.optims.optimizer)
                grad_norm_dict = self.gradient_clipper(model=self.model)
                for key, grad_norm in grad_norm_dict.items():
                    if isnan(grad_norm):
                        raise ValueError(f"{key} has NaN gradient. Stopping training.")
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

        loss_names = [f"Loss/val_{name}" for name in self.val_metrics_to_log if "loss" in name]
        loss_trackers = {name: MetricTracker(name) for name in loss_names}

        score_names = [f"Eval/val_{name}" for name in self.val_metrics_to_log if "score" in name]
        score_trackers = {name: MetricTracker(name) for name in score_names}

        performance_trackers = list(loss_trackers.values()) + list(score_trackers.values())

        amp_type = torch.bfloat16 if self.optim_cfg.amp.amp_dtype == "bfloat16" else torch.float16

        # Validate on each validation dataset (separated unlike training)
        for dataset_name, val_loader in self.val_loader.items():
            # if debugging
            max_iters = len(val_loader) if self.max_val_iters is None else self.max_val_iters

            # Track performance per dataset
            for tracker in performance_trackers:
                tracker.reset()

            # Set epoch for wai dataset samplers 
            # (set to 0 so the order is the same every time for visualization)
            if hasattr(val_loader, "dataset") and hasattr(val_loader.dataset, "set_epoch"):
                val_loader.dataset.set_epoch(0)
            if hasattr(val_loader, "sampler") and hasattr(val_loader.sampler, "set_epoch"):
                val_loader.sampler.set_epoch(0)
            if hasattr(val_loader, "batch_sampler") and hasattr(val_loader.batch_sampler, "set_epoch"):
                val_loader.batch_sampler.set_epoch(0)
            
            batch_start_time = time.time()
            
            last_batch = None
            last_preds = None
            for data_iter, batch in enumerate(val_loader):
                # if debugging
                if data_iter >= max_iters:
                    break

                # Measure data loading time
                data_time.update(time.time() - batch_start_time)
                
                # Process batch
                mapa_batch = batch  # keep for benchmarking
                batch = self._process_batch(batch)
                batch = move_data_to_device(batch, self.device, non_blocking=True)  # to device
                last_batch = batch
                
                # Forward pass
                with torch.no_grad():
                    with torch.autocast(
                        device_type=self.device.type,
                        enabled=self.optim_cfg.amp.enabled,
                        dtype=amp_type,
                    ):
                        _, last_preds = self._step(batch, "val", loss_trackers)

                self.steps["val"] += 1
                
                # Benchmark on validation set
                if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    mapa_preds = self.model.module.convert_preds_to_mapa(last_preds)
                else:
                    mapa_preds = self.model.convert_preds_to_mapa(last_preds)
                self.benchmark(mapa_batch, mapa_preds, score_trackers)

                # Track batch inference time
                batch_time.update(time.time() - batch_start_time)
                batch_start_time = time.time()

                # Memory tracking
                if torch.cuda.is_available():
                    mem.update(torch.cuda.max_memory_allocated() // 1e9)
            
            # Sync metrics across processes
            for tracker in performance_trackers:
                tracker.synchronize_between_processes()

            # Log metrics for dataset
            if self.rank == 0:
                # Log to console
                progress_str = f"Val Epoch {self.epoch + 1} ({dataset_name}) - AVG"
                loss_dict = {tracker.metric_name: tracker.average for tracker in loss_trackers.values()}
                loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items())
                logging.info("%s %s", progress_str, loss_str)
                score_dict = {tracker.metric_name: tracker.average for tracker in score_trackers.values()}
                score_str = " | ".join(f"{k}: {v:.4f}" for k, v in score_dict.items())
                logging.info("%s %s", progress_str, score_str)

                # Gather metrics
                val_losses = {f"{name}_{dataset_name}": tracker.average for name, tracker in loss_trackers.items()}
                val_scores = {f"{name}_{dataset_name}": tracker.average for name, tracker in score_trackers.items()}

                # Gather visuals
                # NOTE: only logging last item of last batch of main process
                depth_vis, pred_ptc, gt_ptc = self.visualize_preds(last_batch, last_preds)
                
                # -- Log to wandb
                log_dict = {
                    **val_losses, 
                    **val_scores,
                }

                log_vis_freq = self.log_cfg.log_vis_freq or 1
                # only add ground-truth point cloud on first epoch 
                if self.epoch == 0 and log_vis_freq <= self.max_epochs:
                    log_dict.update({f"PointClouds/{dataset_name}_gt": gt_ptc})
                # only log visuals every vis_freq epochs to avoid storage misuse
                if (self.epoch + 1) % log_vis_freq == 0:
                    log_dict.update({
                        f"Depths/{dataset_name}": depth_vis, 
                        f"PointClouds/{dataset_name}_pred": pred_ptc,
                    })

                wandb.log(log_dict, step=self.steps['train'] - 1)

        # Display efficiency across validation
        efficiency_trackers = [batch_time, data_time, mem]
        for tracker in efficiency_trackers:
            tracker.synchronize_between_processes()

        if self.rank == 0:
            progress_str = f"Val Epoch {self.epoch + 1} - AVG"
            metrics_dict = {tracker.metric_name: tracker.average for tracker in efficiency_trackers}
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
                    device_type=self.device.type,
                    enabled=self.optim_cfg.amp.enabled,
                    dtype=amp_type,
                ):
                    # Forward pass
                    loss_dict, _ = self._step(chunked_batch, phase, loss_trackers)
                
                loss = loss_dict["loss_objective"]
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
                # will be updated in _run_steps_on_batch_chunks
                if phase == "train" and key == "loss_objective":
                    continue
                value = loss_dict[key].item() if torch.is_tensor(loss_dict[key]) else loss_dict[key]
                loss_key = f"Loss/{phase}_{key}"
                if loss_key in loss_trackers:
                    loss_trackers[loss_key].update(value, batch_size)
        
        return loss_dict, y_hat

    #----------#
    #   Eval   #
    #----------#
    def benchmark(self, batch: List[Dict], preds: List[Dict], trackers: Dict[str, MetricTracker]):
        """
        Benchmark tasks on validation set

        Args:
            batch: A batch from a WAI dataloader
            preds: Preds converted to MapA format
            tracker: MetricTrackers for each benchmark to run
        """
        for view in batch:
            view["idx"] = view["idx"][2:]

        # Transfer batch to device
        ignore_keys = set([
            "depthmap",
            "dataset",
            "label",
            "instance",
            "idx",
            "true_shape",
            "rng",
            "data_norm_type",
        ])
        for view in batch:
            for name in view.keys():  # pseudo_focal
                if name in ignore_keys:
                    continue
                view[name] = view[name].to(self.device, non_blocking=True)

        eval_metrics = [
            name.replace("Eval/val_score_", "") 
            for name in trackers.keys() 
            if name.startswith("Eval/val_score_")
        ]
        if not eval_metrics:
            return
        scores = compute_results_for_batcb(batch, preds, eval_metrics, reduce_mean=True)

        for name, score in scores.items():
            tracker_name = f"Eval/val_score_{name}"
            trackers[tracker_name].update(score, batch[0]["img"].shape[0])

    #-------------------------#
    #  Logging/Checkpointing  #
    #-------------------------#
    def visualize_preds(self, input_batch: Dict, preds_batch: Dict, batch_idx: int=-1):
        """
        Visualize depth and point maps for batch of predictions 
        Args:
            input_batch (Dict): The input to VGGT
            preds_batch (Dict): The output from VGGT
            batch_idx (int): The sequence in the batch to visualize. Currently
                             only supports visualizing one sequence.
                             Defaults to last.
        Returns:
            depth_visuals (wandb.Image): The gt and pred depthmaps for each image in the sequence
                            wrapped in a wandb Image.
            pred_pt (wandb.Object3D): The predicted pointmap as a wandb 3DObject.
            gt_ptc (wandb.Object3D): The ground truth pointmap as a wandb 3DObject.
        """
        images = input_batch["images"][batch_idx]  # [S, 3, H, W]
        pred_depth = preds_batch["depth"][batch_idx]  # [S, H, W, 1]
        pose_enc = preds_batch["pose_enc"][batch_idx].unsqueeze(0)  # [1, S, 9], pose_encoding_to_extri_intri expects batch dim
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        extrinsic = extrinsic.squeeze(0)  # [S, 3, 4]
        intrinsic = intrinsic.squeeze(0)  # [S, 3, 3]

        if "world_points" in preds_batch:
            pred_world_points = preds_batch["world_points"][batch_idx].cpu().numpy()  # [S, H, W, 3]
            pred_conf = preds_batch["world_points_conf"][batch_idx].cpu().numpy() # [S, H, W]
        else:
            pred_world_points = unproject_depth_map_to_point_map(pred_depth, extrinsic, intrinsic)
            pred_conf = preds_batch["depth_conf"][batch_idx].cpu().numpy()
        
        gt_ptc, pred_ptc = create_point_cloud_visualization(
            gt_world_points=input_batch["world_points"][batch_idx].cpu().numpy(),
            pred_world_points=pred_world_points,
            pred_conf=pred_conf,
            valid_masks=input_batch["point_masks"][batch_idx].cpu().numpy(),
            images=images.cpu().numpy(),
        )
        
        depth_visuals = create_depth_grid_visualization(
            images=images,
            gt_depths=input_batch["depths"][batch_idx],
            pred_depths=pred_depth.squeeze(),
            valid_masks=input_batch["point_masks"][batch_idx]
        )

        return depth_visuals, pred_ptc, gt_ptc

    def _create_ckpt_name(self, prefix: str):
        patch_embed_conf = self.model_cfg.model_config.patch_embed_config
        if patch_embed_conf:
            model_id = patch_embed_conf.model_id.split('/')[-1]
            prefix += f"-{model_id}"
        return prefix
