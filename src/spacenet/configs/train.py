from dataclasses import dataclass, field, fields, MISSING
from ml_tools.training.schedulers import SchedConfig as CoreSchedConfig
from .general import GeneralConfig

@dataclass
class TrainerConfig:
    epochs: int
    logdir: str
    exp_name: str
    log_interval: int = 10
    val_interval: int = 1
    mode: str = 'pre-event only'  # 'pre-event only' or 'pre-event'
    amp_dtype_str: str = "bfloat16"
    use_amp: bool = False
    freeze_bn: bool = False
    
    @classmethod
    def from_full(cls, full: GeneralConfig) -> "TrainerConfig":
        init_kwargs = {}
        for f in fields(cls):
            name = f.name
            if hasattr(full, name):
                init_kwargs[name] = getattr(full, name)
            else:
                # handle defaults: if no default and not in full, error
                if f.default is MISSING and f.default_factory is MISSING:
                    raise ValueError(f"Missing required config field: {name}")
                # otherwise let dataclass default handle it

        return cls(**init_kwargs)
    
@dataclass
class OptimConfig:
    lr: float
    weight_decay: float
    momentum: float = 0.9
    optimizer_type: str = 'AdamW'
    
    @classmethod
    def from_full(cls, full: GeneralConfig, rename: dict = None) -> "OptimConfig":
        init_kwargs = {}
        for f in fields(cls):
            name = f.name
            if name not in rename.keys():
                rename[name] = name
            if hasattr(full, rename[name]):
                init_kwargs[name] = getattr(full, rename[name])
            else:
                # handle defaults: if no default and not in full, error
                if f.default is MISSING and f.default_factory is MISSING:
                    raise ValueError(f"Missing required config field: {rename[name]}")
                # otherwise let dataclass default handle it

        return cls(**init_kwargs)
    
@dataclass
class SchedConfig(CoreSchedConfig):
    
    @classmethod
    def from_full(cls, full: GeneralConfig, rename: dict = None) -> "SchedConfig":
        init_kwargs = {}
        for f in fields(cls):
            name = f.name
            if name not in rename.keys():
                rename[name] = name
            if name == 'lr_init_factor':
                init_kwargs[name] = full.start_lr / full.peak_lr
            if hasattr(full, rename[name]):
                init_kwargs[name] = getattr(full, rename[name])
            else:
                # handle defaults: if no default and not in full, error
                if f.default is MISSING and f.default_factory is MISSING:
                    raise ValueError(f"Missing required config field: {rename[name]}")
                # otherwise let dataclass default handle it

        return cls(**init_kwargs)
    