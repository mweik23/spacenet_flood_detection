from dataclasses import dataclass, fields, MISSING
from .general import GeneralConfig

#TODO: check arguments of the constructor
@dataclass
class ModelConfig:
    num_classes: int
    base_channels: int
    depth: int
    
    @classmethod
    def from_full(cls, full: GeneralConfig) -> "ModelConfig":
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