from tkinter import Image
import torch
from torch import nn
from torch import optim
from pathlib import Path
import json
from dataclasses import asdict

from ml_tools.utils.distributed import setup_dist, DistInfo, maybe_convert_syncbn, wrap_like_ddp
from ml_tools.training.optimizer import get_optimizer


from spacenet.dataset.datasets import get_dataloaders, get_paths, get_channels
from spacenet.utils.cli import build_parser
from spacenet.utils.model_utils import get_param_groups

from spacenet.utils.io import init_run, load_ckp, TrainerConfig, DataConfig, ModelConfig, OptimConfig, SchedConfig
from spacenet.models.UNet_basic import UNet

PROJECT_ROOT = Path(__file__).parent.parent



def main(argv=None):
    #get arguments
    parser = build_parser()
    args = parser.parse_args(argv)
    #set up distributed training
    dist_info = setup_dist(arg_num_workers=args.num_workers)
    if dist_info.is_primary:
        print(dist_info)
        
    dist_rt = DistRuntime.from_dist_info(dist_info)
    
    full_cfg = init_run(args, dist_rt, PROJECT_ROOT)
    
    trainer_cfg = TrainerConfig.from_full(full_cfg)
    data_cfg    = DataConfig.from_full(full_cfg)
    model_cfg   = ModelConfig.from_full(full_cfg)
    optim_cfg   = OptimConfig.from_full(full_cfg, rename={'lr': 'peak_lr'})
    sched_config = SchedConfig.from_full(full_cfg, rename={'kind': 'sched_kind', 'mode': 'sched_mode'}) #TODO: ensure that these parameters exist in full config
    
    
    path_datasets = get_paths(Path(data_cfg.datadir))

    loaders = get_dataloaders(
        datasets=path_datasets,
        rank=dist_rt.rank,
        world_size=dist_rt.dist_config.world_size,
        seed=full_cfg.seed,
        mode=trainer_cfg.mode,
        **asdict(data_cfg)
    )
    in_channels = get_channels(path_datasets)
    model = UNet(in_channels=in_channels, **asdict(model_cfg))
    model = maybe_convert_syncbn(model, dist_rt.cfg.device_type, dist_rt.cfg.world_size)
    model = model.to(dist_rt.device)
    optimizer = get_optimizer(model, **asdict(optim_cfg))
    
    ddp_model = wrap_like_ddp(model, dist_rt.device, dist_rt.local_rank, use_ddp=(dist_rt.cfg.world_size>1))
    
    if full_cfg.pretrained != '':
        start_epoch = load_ckp(f"{full_cfg.logdir}/{full_cfg.pretrained}/best-val-model.pt",
                               ddp_model,
                               optimizer=optimizer if full_cfg.ld_optim_state else None,
                               device=dist_rt.device)
    else:
        start_epoch = 0
        
    loss_fn = nn.CrossEntropyLoss()