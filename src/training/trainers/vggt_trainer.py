import torch.nn as nn
import torch.distributed as dist
from pathlib import Path
from omegaconf import DictConfig
from typing import Dict

from src.datasets.wai_dataset import build_wai_dataloader
from src.models.vggt import VGGT
from src.training.optimizer import create_optimizer, create_scheduler, OptimizerWrapper
from src.training.losses.multitask_loss import MultitaskLoss
from src.training.gradient_clip import GradientClipper
from src.training.utils import *
from .trainer import Trainer

class VGGTTrainer(Trainer):
    """
    Base DDP trainer class
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    # Setup datasets
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
    
    # Setup model
    def _setup_models(self):
        # initialize VGGT from official checkpoint or model config
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

    # Setup loss / criterion
    def _setup_loss(self):
        self.loss = MultitaskLoss(**self.loss_cfg.loss_config)

    # Setup optimizer
    def _setup_optimizer(self):
        if self.optim_cfg.gradient_clip:
            self.gradient_clipper = GradientClipper(self.optim_cfg.gradient_clip.module_configs)
        if self.self.optim_cfg.amp.enabled:
            self.scaler = torch.amp.GradScaler(device=self.device.type)

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
