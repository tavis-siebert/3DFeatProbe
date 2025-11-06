import os
import logging
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

def init_distributed(backend: str = None):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        distributed = True
    else:
        logger.info("Not using distributed mode")
        dist_rank, local_rank, world_size = 0, 0, 1
        distributed = False
    
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
        dist.barrier()

    return dist_rank, local_rank, world_size, distributed

