from dataclasses import dataclass, fields, MISSING
from pathlib import Path
from spacenet.dataset.data_utils import get_num_classes
from .general import GeneralConfig
from typing import Optional

#TODO: check arguments of the constructor
@dataclass
class ModelConfig:
    num_classes: int
    base_channels: int
    depth: int
    in_channels: int
    
    @classmethod
    def from_full(cls, full: GeneralConfig, **kwargs) -> "ModelConfig":
        init_kwargs = {}
        init_kwargs.update(kwargs)
        for f in fields(cls):
            name = f.name
            if name == 'num_classes':
                init_kwargs[name] = get_num_classes(Path(full.datadir))
            elif name not in init_kwargs:
                if hasattr(full, name):
                    init_kwargs[name] = getattr(full, name)
                else:
                    # handle defaults: if no default and not in full, error
                    if f.default is MISSING and f.default_factory is MISSING:
                        raise ValueError(f"Missing required config field: {name}")
                    # otherwise let dataclass default handle it

        return cls(**init_kwargs)