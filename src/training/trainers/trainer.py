import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import DictConfig

from src.training.distributed import init_distributed
from src.utils.logging import setup_logging
from src.training.utils import *

# TODO: add checks for null configs on necessary items  

class Trainer:
    """
    Base trainer class
    """
    def __init__(self, cfg: DictConfig):
        # Separate top level configs
        self.dataset_cfg = cfg.dataset
        self.model_cfg = cfg.model
        self.train_cfg = cfg.training

        # Device and DDP Params
        self.dist_cfg = self.train_cfg.distributed
        self.rank, self.local_rank, self.world_size, self.distributed = init_distributed(self.dist_cfg.backend)
        self._setup_device()

        # Checkpointing
        self.ckpt_cfg = self.train_cfg.checkpoint
        self._setup_checkpointing(self.ckpt_cfg)

        # Logging
        self.log_cfg = self.train_cfg.logging
        self._setup_logging(self.log_cfg)

        # Dataset and Dataloaders
        self.train_loader, self.val_loader = None, None
        self._setup_datasets(self.dataset_cfg, self.train_cfg)

        # Model
        self.model = None
        self._setup_model(self.model_cfg)

        # Loss
        self.loss = None
        self.loss_cfg = self.train_cfg.loss
        self._setup_loss(self.loss_cfg)

        # Optimizer
        self.optims = None
        self.optim_cfg = self.train_cfg.optim
        self._setup_optimizer(self.optim_cfg)

        # Training params
        self.seed = self.train_cfg.seed
        set_seeds(self.seed, self.rank)

        self.accum_steps = self.train_cfg.accum_steps
        self.max_epochs = self.train_cfg.max_epochs
        self.epoch = 0
        self.steps = {"train": 0, "val": 0}

        self.eval_freq = self.train_cfg.eval_freq

        # Load last checkpoint
        if self.ckpt_cfg.resume_checkpoint_path:
            self._load_from_checkpoint(self.ckpt_cfg.resume_checkpoint_path)
        
        # DDP model
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else [],
                **self.dist_cfg.ddp_args
            )
            dist.barrier()

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

    # -- Set up logger (supports DDP)
    def _setup_logging(self):
        self.log_dir = self.log_cfg.logging_dir

        _ = setup_logging(
            __name__,
            self.log_dir,
            rank=self.rank
        )

        self.log_freq = self.log_cfg.log_freq
        self.train_metrics_to_log = self.log_cfg.metrics_to_log.train
        self.val_metrics_to_log = self.log_cfg.metrics_to_log.val

    # -- Setup datasets
    def _setup_datasets(self):
        raise NotImplementedError("Must be implemented by subclasses")
    
    # -- Setup model
    def _setup_models(self):
        raise NotImplementedError("Must be implemented by subclasses")

    # -- Setup loss / criterion
    def _setup_loss(self):
        raise NotImplementedError("Must be implemented by subclasses")

    # -- Setup optimizer
    def _setup_optimizer(self):
        raise NotImplementedError("Must be implemented by subclasses")

    # -- Load training state from previous checkpoint
    def _load_from_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info("Loading from checkpoint: ", ckpt_path)
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
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}

        # load AMP scaler state if available
        if self.optim_cfg.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        del checkpoint
    
    #-------------------#
    #   Functionality   #
    #-------------------#
    # -- Training
    def train(self):
        raise NotImplementedError("Must be implemented by subclasses")
