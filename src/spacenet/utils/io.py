import os
import json
import random
import string
import numpy as np
import torch
import shutil
from ml_tools.utils.random import set_global_seed
from spacenet.configs import GeneralConfig
from spacenet.utils.paths import get_project_root
    
    
def make_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)   # remove the directory and all its contents
    os.makedirs(path) 

def init_run(args, dist_runtime, pt_overwrite_keys=(), cfg_extra: dict = None):
    if cfg_extra is None:
        cfg_extra = {}
    project_root = get_project_root()
    if args.seed is None: # use random seed if not specified
        args.seed = np.random.randint(100)
    if args.exp_name == '': # use random strings if not specified
        args.exp_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    set_global_seed(args.seed, rank=dist_runtime.rank)
    
    cfg_extra['logdir'] = str(project_root / args.logdir)
    cfg_extra['datadir'] = str(project_root / args.datadir)
    if args.pretrained != '':
        if '/' in args.pretrained:
            pt_exp = args.pretrained.split('/')[0]
        else:
            pt_exp = args.pretrained
        with open(f"{cfg_extra['logdir']}/{pt_exp}/config.json", 'r') as file:
            pt_args = json.load(file)
        pt_args_overwrite = {k: pt_args[k] for k in pt_overwrite_keys if k in pt_args}
        cfg_extra = {**cfg_extra, **pt_args_overwrite}
    cfg = GeneralConfig.from_args_and_dist(args, dist_runtime.cfg, cfg_extra)
    if (dist_runtime.rank == 0): # master
        make_clean_dir(f"{cfg.logdir}/{args.exp_name}")
        d = cfg.as_dict()
        with open(f"{cfg.logdir}/{args.exp_name}/config.json", 'w') as f:
            json.dump(d, f, indent=4)
            f.close()
    return cfg

def load_ckp(checkpoint_fpath, model, optimizer=None, device=torch.device('cpu')):
    checkpoint = torch.load(checkpoint_fpath, map_location=device, weights_only=True)
    print('initial load of model state dict...')
    incompat = model.load_state_dict(checkpoint['state_dict'], strict=False)
    assert len(incompat.missing_keys) == 0, 'some model keys are missing in loaded state: ' + str(incompat.missing_keys)
    print("all keys in the current model loaded successfully")     # expected: keys for target_* (new modules)
    if len(incompat.unexpected_keys) > 0:
        print('WARNING: some keys in loaded state are not used in the model: ' + str(incompat.unexpected_keys))
    else:
        print('all loaded keys used in the model')
        
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']