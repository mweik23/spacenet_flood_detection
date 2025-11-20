from typing import List
from torch import nn

def get_param_groups(model: nn.Module, lr: float, weight_decay: float) -> List[dict]:
    """Get parameter groups for optimizer."""
    params = model.parameters()
    param_groups = [{
        "params": list(params),
        "lr": lr,
        "weight_decay": weight_decay,
    }]
    total_params = sum(p.numel() for p in params)
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    return param_groups