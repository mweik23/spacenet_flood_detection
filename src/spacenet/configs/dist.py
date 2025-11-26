from dataclasses import dataclass, field, asdict, fields, MISSING
from typing import Optional
import torch
import torch.distributed as dist

from ml_tools.utils.distributed import DistInfo

@dataclass
class DistConfig:
    backend: str
    world_size: int
    master_addr: str
    master_port: int
    device_type: str
    has_cuda: bool
    num_workers: int          # TODO: move to DataConfig
    cpus_per_task: Optional[int] = None
    
    @classmethod
    def from_dist_info(cls, dist_info: DistInfo, rename: dict = None) -> "DistConfig":
        init_kwargs = {}
        if rename is None:
            rename = {}
        for f in fields(cls):
            name = f.name
            if name not in rename.keys():
                rename[name] = name
            if hasattr(dist_info, rename[name]):
                init_kwargs[name] = getattr(dist_info, rename[name])
            else:
                # handle defaults: if no default and not in full, error
                if f.default is MISSING and f.default_factory is MISSING:
                    raise ValueError(f"Missing required config field: {rename[name]}")
                # otherwise let dataclass default handle it

        return cls(**init_kwargs)
    
@dataclass
class DistRuntime:
    cfg: DistConfig
    rank: int
    local_rank: int
    node_rank: int
    is_primary: bool
    device: torch.device
    device_name: str
    
    @property
    def initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def destroy(self) -> None:
        if self.initialized:
            dist.destroy_process_group()
    @classmethod
    def from_dist_info(cls, dist_info: DistInfo, rename: dict = None) -> "DistRuntime":
        init_kwargs = {}
        if rename is None:
            rename = {}
        for f in fields(cls):
            name = f.name
            if name not in rename.keys():
                rename[name] = name
            if name=='cfg':
                init_kwargs[name] = DistConfig.from_dist_info(dist_info)
            elif name=='device':
                init_kwargs[name] = torch.device(getattr(dist_info, 'device_type'))
            else:
                if hasattr(dist_info, rename[name]):
                    init_kwargs[name] = getattr(dist_info, rename[name])
                else:
                    # handle defaults: if no default and not in full, error
                    if f.default is MISSING and f.default_factory is MISSING:
                        raise ValueError(f"Missing required config field: {rename[name]}")
                    # otherwise let dataclass default handle it

        return cls(**init_kwargs)
    