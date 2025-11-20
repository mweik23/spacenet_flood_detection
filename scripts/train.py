from ml_tools.utils.distributed import setup_dist, DistInfo, maybe_convert_syncbn
from ml_tools.training.scheulers import SchedConfig
import torch
from torch import nn
from torch import optim
from spacenet.dataset.datasets import get_dataloaders, get_paths, wrap_like_ddp
from spacenet.utils.cli import build_parser
from spacenet.utils.model_utils import get_param_groups
from spacenet.dataset.collate import TileCollator
from spacenet.utils.io import init_run, load_ckp
from spacenet.models.UNet_basic import UNet
from pathlib import Path
import json
PROJECT_ROOT = Path(__file__).parent.parent



def main(argv=None):
    #get arguments
    parser = build_parser()
    args = parser.parse_args(argv)
    #set up distributed training
    dist_info: DistInfo = setup_dist(arg_num_workers=args.num_workers)
    if dist_info.is_primary:
        print(dist_info)
    device = torch.device(dist_info.device_type)
    cfg = init_run(args, dist_info, PROJECT_ROOT)
    
    if cfg.mode == 'pre-event':
        img_size = 1300 #TODO: infer this from data
    
    collate = TileCollator(img_size=img_size,
                           core_size=cfg.core_size,
                           halo_size=cfg.halo_size,
                           stride=cfg.stride,
                           num_tiles=cfg.num_tiles,
                           random_order=True,
                           num_sets=cfg.num_tile_sets,
                           verbose=False)
    
    path_datasets = get_paths(Path(cfg.datadir)) #TODO: create datasets
    loaders = get_dataloaders(
        datasets=path_datasets,
        batch_size=args.batch_size,
        collate_fn=collate,  # TODO: add collate function
        num_workers=args.num_workers,
        dist_info=dist_info,
    )
    
    if cfg.model_config != '':
        with open(PROJECT_ROOT / 'model_configs' / cfg.model_config, 'r') as f:
            model_config = json.load(f)
    #TODO: get in_channels from data
    model = UNet(in_channels=3, num_classes=cfg.num_classes, base_channels=cfg.base_channels, depth=cfg.depth)
    model = maybe_convert_syncbn(model, dist_info.device_type, dist_info.world_size)
    model = model.to(device)
    param_groups = get_param_groups(model, cfg.peak_lr, cfg.weight_decay)
    optimizer = optim.AdamW(param_groups)
    ddp_model = wrap_like_ddp(model, device, dist_info.local_rank, use_ddp=(dist_info.world_size>1))
    
    if cfg.pretrained != '':
        start_epoch = load_ckp(f"{cfg.logdir}/{cfg.pretrained}/best-val-model.pt",
                               ddp_model,
                               optimizer=optimizer if cfg.ld_optim_state else None,
                               device=device,
                               use_target_model=len(cfg.target_model_groups)>0)
    else:
        start_epoch = 0
        
    for g in optimizer.param_groups:
        g.setdefault('initial_lr', g['lr'])
        
    sched_config = SchedConfig(
        kind = "warmup_plateau",
        lr_min=cfg.start_lr/cfg.peak_lr,
        warmup_epochs = cfg.warmup_epochs,
        mode = "min",
        factor = cfg.reduce_factor,
        patience = cfg.patience,
        threshold = cfg.threshold
    )
    
    loss_fn = nn.CrossEntropyLoss()