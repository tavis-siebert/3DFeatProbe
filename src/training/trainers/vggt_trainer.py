import torch.nn as nn
import torch.distributed as dist
from pathlib import Path
import time
import numpy as np
from omegaconf import DictConfig
from typing import Dict, Union
import wandb

from src.datasets.wai_dataset import build_wai_dataloader
from src.models.vggt import VGGT
from src.utils.camera import invert_pose
from src.training.optimizer import create_optimizer, create_scheduler, OptimizerWrapper
from src.training.losses.multitask_loss import MultitaskLoss
from src.training.gradient_clip import GradientClipper
from src.training.utils import *
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
    def _setup_models(self):
        # initialize VGGT from official checkpoint or model config
        logging.info("Initializing model")
        if self.model_cfg.load_pretrained:
            self.model = VGGT.from_pretrained()
        else:
            model_args = self.model_cfg.model_config
            # check img_size = rain_resolution
            train_h, train_w = self.train_cfg.resolution.train
            if train_h != train_w:
                raise ValueError(f"VGGT must be trained with square images. Got {(train_h, train_w)}")
            if train_h != model_args.img_size:
                logging.warning(f"Got train image size {train_h} and config image size {model_args.img_size}. Defaulting to training size")
                model_args.img_size = train_h

            self.model = VGGT(**model_args)
        logging.info(f"Initialized Model: {str(self.model)}")

        # swap out patch embed
        patch_embed_ckpt_path = self.model_cfg.model_config.pretrained_patch_embed_path
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
        logging.info(f"Setting loss function {self.loss_cfg}")
        self.loss = MultitaskLoss(**self.loss_cfg.loss_config)

    # -- Setup optimizer
    def _setup_optimizer(self):
        if self.optim_cfg.gradient_clip:
            self.gradient_clipper = GradientClipper(self.optim_cfg.gradient_clip.module_configs)
            logging.info("Gradient clipping set")
        if self.self.optim_cfg.amp.enabled:
            self.scaler = torch.amp.GradScaler(device=self.device.type)
            logging.info("AMP scaling set")

        # freeze submodules
        if self.optim_cfg.frozen_submodules:
            logging.info(f"Freezing Submodules {self.optim_cfg.frozen.submodules}") 
            self.model = freeze_modules(
                self.model, patterns=self.optim_cfg.frozen_submodules
            )

        # construct optimizer and schedulers
        optimizer_cfg = self.optim_cfg.optimizer
        logging.info(f"Initializing Optimizer: {optimizer_cfg}")
        optimizer = create_optimizer(self.model, optimizer_cfg.name, **optimizer_cfg.optimizer_config)
        
        lr_sched_cfg, wd_sched_cfg = self.optim_cfg.schedulers.lr, self.optim_cfg.schedulers.weight_decay
        schedulers = {}
        if lr_sched_cfg:
            logging.info(f"Initializing LR Scheduler {lr_sched_cfg}")
            lr_scheduler = create_scheduler(lr_sched_cfg.name, **lr_sched_cfg.scheduler_config)
            schedulers["lr"] = lr_scheduler
        if wd_sched_cfg:
            logging.info(f"Initializing Weight Decay Scheduler {wd_sched_cfg}")
            wd_scheduler = create_scheduler(wd_sched_cfg.name, **wd_sched_cfg.scheduler_config)
            schedulers["weight_decay"] = wd_scheduler

        self.optims = OptimizerWrapper(optimizer, schedulers)

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
        def __convert_numpy(arr: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
            if isinstance(arr, np.ndarray):
                return torch.from_numpy(arr)
            return arr

        image_batch, depth_batch, valid_mask_batch = [], [], []
        intrinsics_batch, extrinsics_batch = [], []
        world_pts_batch = []

        for view in views:
            # rgb image
            image = view['img'] # [B, 3, H, W]
            image_batch.append(__convert_numpy(image))

            # depth map
            depthmap = view['depthmap']  # [B, H, W]
            depth_batch.append(__convert_numpy(depthmap))

            # mask
            valid_mask = view['valid_mask']  # [B, H, W]
            valid_mask_batch.append(__convert_numpy(valid_mask))

            # camera intrinsics
            intrinsics = view['camera_intrinsics']  # [B, 3, 3]
            intrinsics_batch.append(__convert_numpy(intrinsics))

            # camera pose (ggt expects world2cam)
            cam2world = __convert_numpy(view['camera_pose']) # [B, 4, 4]
            world2cam = invert_pose(cam2world)
            pose = world2cam[:, :3, :]  # [B, 3, 4]
            extrinsics_batch.append(pose)

            # point maps
            pts3d = view['pts3d']  # [B, H, W, 3]
            world_pts_batch.append(__convert_numpy(pts3d))

        # stack and arrange as (B, num_views, ...)
        image_batch = torch.stack(image_batch, dim=1)
        depth_batch = torch.stack(depth_batch, dim=1)
        valid_mask_batch = torch.stack(valid_mask_batch, dim=1)
        intrinsics_batch = torch.stack(intrinsics_batch, dim=1)
        extrinsics_batch = torch.stack(extrinsics_batch, dim=1)
        world_pts_batch = torch.stack(world_pts_batch, dim=1)
        
        return {
            "images": image_batch,
            "extrinsics": extrinsics_batch,
            "intrinsics": intrinsics_batch,
            "depths": depth_batch,
            "world_points": world_pts_batch,
            "point_masks": valid_mask_batch,
        }

    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process batch with normalization (from VGGT trainer)"""
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
        """Main training loop"""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed + self.epoch * 100, self.rank)
            
            self.train_epoch()
            self.save_checkpoint(self.epoch)

            if (self.epoch + 1) % self.eval_freq == 0:
                self.val_epoch()
            
            self.epoch += 1

    #TODO
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        # TODO: Setup meters using verbose logging
        # batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        # data_time = AverageMeter("Data Time", self.device, ":.4f")
        # mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        
        loss_names = [f"Loss/train_{name}" for name in self.train_metrics_to_log]
        # loss_meters = {name: AverageMeter(name, self.device, ":.4f") for name in loss_names}
        
        # Gradient clipping meters
        # if hasattr(self, 'gradient_clipper') and self.gradient_clipper:
        #     for config in self.gradient_clipper.configs:
        #         param_names = ",".join(config['module_names'])
        #         loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")
        
        # progress = ProgressMeter(
        #     num_batches=len(self.train_loader),
        #     meters=[
        #         batch_time,
        #         data_time,
        #         mem,
        #         self.time_elapsed_meter,
        #         *loss_meters.values(),
        #    ],
        #     real_meters={},
        #     prefix=f"Train Epoch: [{self.epoch}]",
        # )
        
        end = time.time()
        
        # Setup gradient clipping
        if hasattr(self, 'gradient_clipper') and self.gradient_clipper:
            self.gradient_clipper.setup_clipping(self.model)
        
        for data_iter, batch in enumerate(self.train_loader):
            #TODO: Measure data loading time if verbose logging used
            # data_time.update(time.time() - end)
            
            # Convert batch format from map-anything to VGGT
            batch = self._convert_mapa_batch_to_vggt(batch)
            
            # Process batch (normalization)
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            
            # Move to device
            batch = copy_data_to_device(batch, self.device, non_blocking=True)
            
            # Gradient accumulation chunking
            if self.accum_steps == 1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, self.accum_steps)
            
            # Run forward/backward
            self._run_steps_on_batch_chunks(chunked_batches, "train", loss_meters)
            
            # Optimizer step
            if (data_iter + 1) % self.accum_steps == 0:
                # Gradient clipping
                if hasattr(self, 'gradient_clipper') and self.gradient_clipper:
                    for optim in [self.optims]:
                        self.scaler.unscale_(optim.optimizer)
                    grad_norm_dict = self.gradient_clipper(model=self.model)
                    for key, grad_norm in grad_norm_dict.items():
                        loss_meters[f"Grad/{key}"].update(grad_norm)
                
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
                
                # Log to wandb
                if data_iter % self.log_freq == 0 and self.rank == 0:
                    log_dict = {
                        "train/epoch": self.epoch,
                        "train/step": self.steps["train"],
                        "train/lr": self.optims.optimizer.param_groups[0]["lr"],
                        "train/where": self.where,
                    }
                    for name, meter in loss_meters.items():
                        log_dict[f"train/{name}"] = meter.avg
                    if self.log_cfg.use_wandb:
                        wandb.log(log_dict, step=self.steps["train"])
            
            #TODO: Memory tracking
            # if torch.cuda.is_available():
            #     mem.update(torch.cuda.max_memory_allocated() // 1e9)
            # batch_time.update(time.time() - end)
            # self.time_elapsed_meter.update(time.time() - self.start_time)
            # end = time.time()
            
            #TODO: Progress display
            # if data_iter % self.log_freq == 0:
            #     progress.display(data_iter)

    #TODO
    def val_epoch(self):
        """Validation epoch"""
        if self.val_loader is None:
            return
        
        self.model.eval()
        
        # Setup meters
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        
        loss_names = [f"Loss/val_{name}" for name in self.val_metrics_to_log]
        loss_meters = {name: AverageMeter(name, self.device, ":.4f") for name in loss_names}
        
        progress = ProgressMeter(
            num_batches=sum(len(loader) for loader in self.val_loader.values()),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix=f"Val Epoch: [{self.epoch}]",
        )
        
        end = time.time()
        
        amp_type = torch.bfloat16 if self.optim_cfg.amp.amp_dtype == "bfloat16" else torch.float16
        
        # Validate on each validation dataset
        for dataset_name, val_loader in self.val_loader.items():
            for data_iter, batch in enumerate(val_loader):
                data_time.update(time.time() - end)
                
                # Convert batch format
                batch = self._convert_mapanything_batch_to_vggt_format(batch)
                
                # Process batch
                with torch.cuda.amp.autocast(enabled=False):
                    batch = self._process_batch(batch)
                
                batch = copy_data_to_device(batch, self.device, non_blocking=True)
                
                # Forward pass
                with torch.no_grad():
                    with torch.cuda.amp.autocast(
                        enabled=self.optim_cfg.amp.enabled,
                        dtype=amp_type,
                    ):
                        loss_dict = self._step(batch, self.model, "val", loss_meters)
                
                # Log to wandb
                if data_iter % self.log_freq == 0 and self.rank == 0:
                    log_dict = {
                        f"val/{dataset_name}/epoch": self.epoch,
                        f"val/{dataset_name}/step": self.steps["val"],
                    }
                    for key, value in loss_dict.items():
                        if torch.is_tensor(value):
                            log_dict[f"val/{dataset_name}/{key}"] = value.item()
                    if self.log_cfg.use_wandb:
                        wandb.log(log_dict, step=self.steps["val"])
                
                batch_time.update(time.time() - end)
                if torch.cuda.is_available():
                    mem.update(torch.cuda.max_memory_allocated() // 1e9)
                end = time.time()
                
                if data_iter % self.log_freq == 0:
                    progress.display(data_iter)

    #TODO
    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Dict],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """Run forward/backward on batch chunks for gradient accumulation"""
        self.optims.optimizer.zero_grad(set_to_none=True)
        
        accum_steps = len(chunked_batches)
        amp_type = torch.bfloat16 if self.optim_cfg.amp.amp_dtype == "bfloat16" else torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            # DDP no_sync for gradient accumulation
            ddp_context = (
                self.model.no_sync() if i < accum_steps - 1 else contextlib.nullcontext()
            )
            
            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_cfg.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(chunked_batch, self.model, phase, loss_meters)
                
                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]
                
                # Check for NaN/Inf
                if not math.isfinite(loss.item()):
                    logging.error(f"Loss is {loss.item()}, stopping training")
                    return
                
                # Scale loss for accumulation
                loss /= accum_steps
                
                # Backward pass
                if self.optim_cfg.amp.enabled:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                loss_meters[loss_key].update(loss.item(), batch_size)

    #TODO
    def _step(self, batch: Dict, model: nn.Module, phase: str, loss_meters: Dict[str, AverageMeter]):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
        """
        # Forward pass - VGGT expects images as keyword arg
        y_hat = model(images=batch["images"])
        
        # Loss computation - MultitaskLoss expects (predictions, batch)
        loss_dict = self.loss(y_hat, batch)
        
        # Update meters
        batch_size = batch["images"].shape[0]
        for key in self.train_metrics_to_log if phase == "train" else self.val_metrics_to_log:
            if key in loss_dict:
                value = loss_dict[key].item() if torch.is_tensor(loss_dict[key]) else loss_dict[key]
                meter_key = f"Loss/{phase}_{key}"
                if meter_key in loss_meters:
                    loss_meters[meter_key].update(value, batch_size)
        
        self.steps[phase] += 1
        return loss_dict
