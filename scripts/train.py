from tkinter import Image
import torch
from torch import nn
from pathlib import Path
from dataclasses import asdict

from ml_tools.utils.distributed import setup_dist, maybe_convert_syncbn, wrap_like_ddp
from ml_tools.training.optimizer import get_optimizer
from ml_tools.training.schedulers import make_scheduler
from ml_tools.utils.random import make_and_set_seed


from spacenet.dataset.datasets import get_dataloaders, get_paths
from spacenet.dataset.data_utils import get_channels
from spacenet.utils.cli import build_parser
from spacenet.utils.io import init_run, load_ckp
from spacenet.configs import TrainerConfig, DataConfig, ModelConfig, OptimConfig, SchedConfig, DistRuntime
from spacenet.models.UNet_basic import UNet
from spacenet.training.trainer import Trainer

def main(argv=None):
    #get arguments
    parser = build_parser()
    args = parser.parse_args(argv)
    
    #set up distributed training
    dist_info = setup_dist(arg_num_workers=args.num_workers)
    dist_rt = DistRuntime.from_dist_info(dist_info)
    
    #initialize full configuration
    full_cfg = init_run(args, dist_rt)
    
    #set up data loaders
    #set seed for setup phase
    make_and_set_seed(args.seed, phase='collate', rank=dist_rt.rank)
    data_cfg = DataConfig.from_full(full_cfg)
    path_datasets = get_paths(Path(data_cfg.datadir), num_data=data_cfg.num_data)
    dataloaders = get_dataloaders(
        datasets=path_datasets,
        rank=dist_rt.rank,
        world_size=dist_rt.cfg.world_size,
        seed=full_cfg.seed,
        mode=full_cfg.mode,
        persistent_workers=data_cfg.num_workers>0,
        pin_memory=True,
        **asdict(data_cfg)
    )
    
    #set up model
    model_cfg = ModelConfig.from_full(
        full_cfg, 
        in_channels=get_channels(Path(path_datasets['train'][0]['pre-event image']))
    )
    make_and_set_seed(args.seed, phase='model', rank=dist_rt.rank)
    model = UNet(**asdict(model_cfg))
    model = maybe_convert_syncbn(model, dist_rt.cfg.device_type, dist_rt.cfg.world_size)
    model = model.to(dist_rt.device)
    ddp_model = wrap_like_ddp(model, dist_rt.device, dist_rt.local_rank, use_ddp=(dist_rt.cfg.world_size>1))
    
    #set up optimizer
    optim_cfg   = OptimConfig.from_full(full_cfg, rename={'lr': 'peak_lr'})
    optimizer = get_optimizer(model, **asdict(optim_cfg))
    
    #set up scheduler
    sched_config = SchedConfig.from_full(full_cfg, rename={'kind': 'sched_kind', 'mode': 'sched_mode'}) #TODO: ensure that these parameters exist in full config
    scheduler = make_scheduler(optimizer, sched_config)
    
    #load from checkpoint if specified
    if full_cfg.pretrained != '':
        last_epoch = load_ckp(f"{full_cfg.logdir}/{full_cfg.pretrained}/best-val-model.pt",
                               ddp_model,
                               optimizer=optimizer if full_cfg.ld_optim_state else None,
                               device=dist_rt.device)
    else:
        last_epoch = -1
    start_epoch = last_epoch + 1
    #set up trainer
    loss_fn = nn.BCEWithLogitsLoss()
    trainer_cfg = TrainerConfig.from_full(full_cfg)
    trainer = Trainer(
        cfg=trainer_cfg,
        model=ddp_model,
        optimizer=optimizer,
        dist_rt=dist_rt,
        scheduler=scheduler,
        loss_fn=loss_fn,
        dataloaders=dataloaders,
        start_epoch=start_epoch,
        base_seed=full_cfg.seed
    )

    #train and test
    if not full_cfg.test_mode:
        trainer.train()
    if full_cfg.test_seed is not None:
        trainer.base_seed = full_cfg.test_seed
    trainer.test()
    
    #clean up distributed training
    if dist_rt.initialized():
        dist_rt.destroy()
        
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    main()

   
    
    
    