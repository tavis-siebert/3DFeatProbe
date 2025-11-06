import torch
from fvcore.common.param_scheduler import *
from typing import Dict, Optional

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

SCHEDULERS = {
    'constantparamscheduler': ConstantParamScheduler, 
    'cosineparamscheduler': CosineParamScheduler, 
    'exponentialparamscheduler': ExponentialParamScheduler, 
    'linearparamscheduler': LinearParamScheduler, 
    'compositeparamscheduler': CompositeParamScheduler, 
    'multistepparamscheduler': MultiStepParamScheduler, 
    'stepparamscheduler': StepParamScheduler, 
    'stepwithfixedgammaparamscheduler': StepWithFixedGammaParamScheduler, 
    'polynomialdecayparamscheduler': PolynomialDecayParamScheduler
}

def create_optimizer(
    model: torch.nn.Module,
    name: str,
    lr: float,
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Returns a torch.optim instance

    Args:
        model (torch.nn.Module): the model whose parameters are optimized
        name (str): the name of the optimizer (e.g. 'AdamW')
        lr (float): optimizer learning rate
        weight_decay (float): optimizer weight decay. Default = 0.0
        **kwargs: other optimizer-specific args (please refer to `torch.optim` documentation) 
    """
    name = name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {name}")
    params = [p for p in model.parameters() if p.requires_grad]
    return OPTIMIZERS[name](params, lr=lr, weight_decay=weight_decay, **kwargs)

def create_scheduler(
    name: str,
    scheduler_config: Dict,
) -> ParamScheduler:
    name = name.lower()
    if name not in SCHEDULERS:
        raise ValueError(f"Unsupported scheduler: {name}")
    
    if name == "compositeparamscheduler":
        scheduler_objs = scheduler_config.get("schedulers")
        if not scheduler_objs: 
            raise KeyError("Expected key 'schedulers' when using CompositeParamScheduler")

        schedulers = []
        for i, sub_scheduler in enumerate(scheduler_objs):
            name, cfg = sub_scheduler.get("name"), sub_scheduler.get("scheduler_config")
            if not (name and cfg): 
                raise KeyError(f"Expected keys 'name' and 'scheduler_config' for scheduler object {i}")
            
            schedulers.append(
                create_scheduler(name, cfg)
            )
        
        lengths, interval_scaling = scheduler_config.get("lengths", None), scheduler_config.get("interval_scaling", None)
        if not (lengths and interval_scaling): 
            raise KeyError("Expected keys 'lengths' and 'interval_scaling'")

        return SCHEDULERS[name](schedulers, lengths, interval_scaling)
    else:
        return SCHEDULERS[name](**scheduler_config)

class OptimizerWrapper:
    """Wrap an optimizer and per-option fvcore schedulers (e.g., lr, weight_decay)."""

    def __init__(self, optimizer: torch.optim.Optimizer, schedulers: Optional[Dict[str, ParamScheduler]] = None):
        self.optimizer = optimizer
        self.schedulers = schedulers or {}
        # Initialize optimizer params to start values
        self.step_schedulers(0.0)

    def step(self, where: float, closure=None):
        self.step_schedulers(where)
        return self.optimizer.step(closure)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def step_schedulers(self, where: float):
        for option, scheduler in self.schedulers.items():
            for group in self.optimizer.param_groups:
                group[option] = scheduler(where)

