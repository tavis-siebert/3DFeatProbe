import os
import wandb
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import DictConfig

from src.training.distributed import init_distributed
from src.training.gradient_clip import GradientClipper
from src.training.optimizer import create_optimizer, create_scheduler, OptimizerWrapper
from src.utils.io import safe_makedirs
from src.utils.logging import setup_logging
from src.training.utils import *

class Trainer:
    """
    Base trainer class
    """
    def __init__(self, cfg: DictConfig):
        # Separate top level configs
        self.cfg = cfg
        self.dataset_cfg = self.cfg.dataset
        self.model_cfg = self.cfg.model
        self.train_cfg = self.cfg.training

        # Device, DDP Params
        self.dist_cfg = self.train_cfg.distributed
        self.rank, self.local_rank, self.world_size, self.distributed = init_distributed(self.dist_cfg.backend)
        
        # Logging
        self.log_cfg = self.train_cfg.logging
        self._setup_logging()

        # Device and seed
        self._setup_device()

        self.seed = self.train_cfg.seed
        set_seeds(self.seed, self.rank)

        # Checkpointing
        self.ckpt_cfg = self.train_cfg.checkpoint
        self._setup_checkpointing()

        # Dataset and Dataloaders
        self.train_loader, self.val_loader = None, None
        logging.info("Setting up train and validation sets")
        self._setup_datasets()

        # Model
        self.model = None
        logging.info(f"Setting up model {self.model_cfg}")
        self._setup_model()

        # Loss
        self.loss = None
        self.loss_cfg = self.train_cfg.loss
        logging.info(f"Setting up loss function: {self.loss_cfg}")
        self._setup_loss()

        # Optimizer
        self.optims = None
        self.optim_cfg = self.train_cfg.optim
        self._setup_optimizer()

        # Training params
        self.accum_steps = self.train_cfg.accum_steps
        self.max_epochs = self.train_cfg.max_epochs
        self.max_train_iters = self.train_cfg.max_train_iters
        self.max_val_iters = self.train_cfg.max_val_iters

        self.epoch = 0
        self.steps = {"train": 0, "val": 0}

        logging.info(f"Model set to train until epoch {self.max_epochs}")
        if self.max_train_iters is not None: logging.info(f"DEBUG MODE: Each train epoch is {self.max_train_iters} iters")
        if self.max_val_iters is not None: logging.info(f"DEBUG MODE: Each val epoch is {self.max_val_iters} iters")
        logging.info(f"Using {self.accum_steps} gradient accumulation steps")

        self.eval_freq = self.train_cfg.eval_freq

        # Load last checkpoint
        if self.ckpt_cfg.resume_checkpoint_path:
            self.load_from_checkpoint(self.ckpt_cfg.resume_checkpoint_path)
        
        # DDP model
        if self.distributed:
            logging.info("Wrapping model with DDP")
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else [],
                **self.dist_cfg.ddp_args
            )
            dist.barrier()
        
        logging.info("Trainer ready to train!")

    #-------------------#
    #       Setup       #
    #-------------------#
    # -- Set device
    def _setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")
        logging.info(f"Device set to {self.device.type}")
    
    # -- Set up checkpointing
    def _setup_checkpointing(self):
        self.ckpt_dir = self.ckpt_cfg.checkpoints_dir
        self.save_freq = self.ckpt_cfg.save_freq
        logging.info(f"Checkpoints will be saved under {self.ckpt_dir} every {self.save_freq} epoch(s)")

    # -- Set up logger (supports DDP)
    def _setup_logging(self):
        # log to file only from main process when using logging lib
        self.log_dir = self.log_cfg.logging_dir
        setup_logging(
            __name__,
            self.log_dir,
            rank=self.rank
        )

        if self.rank == 0:
            wandb.init(
                entity=self.log_cfg.wandb.entity,
                project=self.log_cfg.wandb.project,
                config=self.cfg
            )

        # logging params
        self.log_freq = self.log_cfg.log_freq
        self.train_metrics_to_log = self.log_cfg.metrics_to_log.train
        self.val_metrics_to_log = self.log_cfg.metrics_to_log.val

    # -- Setup datasets
    def _setup_datasets(self):
        raise NotImplementedError("Must be implemented by subclasses")
    
    # -- Setup model
    def _setup_model(self):
        raise NotImplementedError("Must be implemented by subclasses")

    # -- Setup loss / criterion
    def _setup_loss(self):
        raise NotImplementedError("Must be implemented by subclasses")

    # -- Setup optimizer
    def _setup_optimizer(self):
        if self.optim_cfg.gradient_clip:
            self.gradient_clipper = GradientClipper(self.optim_cfg.gradient_clip.module_configs)
            logging.info("Gradient clipping set")
        if self.optim_cfg.amp.enabled:
            self.scaler = torch.amp.GradScaler(device=self.device.type)
            logging.info("AMP scaling set")

        # freeze submodules
        if self.optim_cfg.frozen_submodules:
            logging.info(f"Freezing Submodules {self.optim_cfg.frozen_submodules}") 
            self.model = freeze_modules(
                self.model, patterns=self.optim_cfg.frozen_submodules
            )
            # inform num trainable params
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logging.info(f"Trainable Parameters: {trainable_params:,}")

        # construct optimizer and schedulers
        optimizer_cfg = self.optim_cfg.optimizer
        logging.info(f"Initializing Optimizer: {optimizer_cfg}")
        optimizer = create_optimizer(self.model, optimizer_cfg.name, **optimizer_cfg.optimizer_config)
        
        self.where = 0.0   # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        lr_sched_cfg, wd_sched_cfg = self.optim_cfg.schedulers.lr, self.optim_cfg.schedulers.weight_decay
        schedulers = {}
        if lr_sched_cfg:
            logging.info(f"Initializing LR Scheduler {lr_sched_cfg}")
            lr_scheduler = create_scheduler(lr_sched_cfg.name, dict(lr_sched_cfg.scheduler_config))
            schedulers["lr"] = lr_scheduler
        if wd_sched_cfg:
            logging.info(f"Initializing Weight Decay Scheduler {wd_sched_cfg}")
            wd_scheduler = create_scheduler(wd_sched_cfg.name, dict(wd_sched_cfg.scheduler_config))
            schedulers["weight_decay"] = wd_scheduler

        self.optims = OptimizerWrapper(optimizer, schedulers)

    #-------------------------#
    #  Logging/Checkpointing  #
    #-------------------------#
    # -- Load training state from previous checkpoint
    def load_from_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Loading from checkpoint: {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        # load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=False
        )
        logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        # load optimizer state
        if "optimizer" in checkpoint:
            logging.info(f"Loading optimizer state dict")
            self.optims.optimizer.load_state_dict(checkpoint["optimizer"])

        # load training progress
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        if "steps" in checkpoint:
            self.steps = checkpoint["steps"] 

        # load AMP scaler state if available
        if self.optim_cfg.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        del checkpoint
    
    def save_checkpoint(self, ckpt_name: str, only_model=False):
        """
        Save checkpoint of current training 

        Args:
            ckpt_name (str): the name of the file. Will be saved to `"<self.ckpt_dir>/<ckpt_name>.pt"`
            only_model (bool): whether to save only the model. Default is False.
        """
        # ensure checkpoint dir exists
        safe_makedirs(self.ckpt_dir)
        
        # save model
        model = self.model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module

        checkpoint = {
            "model": model.state_dict()
        }

        # save additional args
        if not only_model:
            checkpoint["epoch"] = self.epoch
            checkpoint["steps"] = self.steps
            checkpoint["optimizer"] = self.optims.optimizer.state_dict()
            if self.optim_cfg.amp.enabled:
                checkpoint["scaler"] = self.scaler.state_dict()

        # write checkpoint to path
        ckpt_path = os.path.join(
            self.ckpt_dir, f"{ckpt_name}.pt"
        )
        logging.info(f"Saving checkpoint at epoch {self.epoch} to {ckpt_path}")
        with open(ckpt_path, "wb") as f:
            torch.save(checkpoint, f)
    
    #-------------------#
    #   Functionality   #
    #-------------------#
    # -- Training
    def train(self):
        raise NotImplementedError("Must be implemented by subclasses")
