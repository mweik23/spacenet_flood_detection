from dataclasses import dataclass, field, fields, MISSING
from .general import GeneralConfig

@dataclass
class CollateConfig:
    core_size: int
    halo_size: int
    stride: int
    num_tiles: int
    num_sets: int
    
    @classmethod
    def from_full(cls, full: GeneralConfig) -> "CollateConfig":
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
class DataConfig:
    collate_cfg: CollateConfig
    datadir: str
    batch_size: int
    num_workers: int = 0
    num_data: int = -1  # use -1 for all data
    
    @classmethod
    def from_full(cls, full: GeneralConfig) -> "DataConfig":
        init_kwargs = {}
        for f in fields(cls):
            name = f.name
            if name == 'collate_cfg':
                init_kwargs[name] = CollateConfig.from_full(full)
            elif hasattr(full, name):
                init_kwargs[name] = getattr(full, name)
            else:
                # handle defaults: if no default and not in full, error
                if f.default is MISSING and f.default_factory is MISSING:
                    raise ValueError(f"Missing required config field: {name}")
                # otherwise let dataclass default handle it

        return cls(**init_kwargs)