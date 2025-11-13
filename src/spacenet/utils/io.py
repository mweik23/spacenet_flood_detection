import os
import json
import random
import string
import numpy as np
import torch
import shutil
from dataclasses import dataclass, field
from spacenet.utils.random import set_global_seed


@dataclass
class TrainingConfig:
    # store everything here dynamically
    _config: dict = field(default_factory=dict)

    @classmethod
    def from_args_and_dist(cls, args, dist_info, extra: dict = None):
        args_dict = vars(args).copy()
        dist_dict = dist_info.shared_dict()
        merged = {**args_dict, **dist_dict, **(extra or {})}
        return cls(merged)

    def __getattr__(self, name):
        if name in self._config:
            return self._config[name]
        raise AttributeError(name)

    def as_dict(self):
        return dict(self._config)

def make_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)   # remove the directory and all its contents
    os.makedirs(path) 

def init_run(args, dist_info, project_root, pt_overwrite_keys= (), cfg_extra: dict = None):
    if args.seed is None: # use random seed if not specified
        args.seed = np.random.randint(100)
    if args.exp_name == '': # use random strings if not specified
        args.exp_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    set_global_seed(args.seed, rank=dist_info.rank)
    
    cfg_extra['log_dir'] = str(project_root / args.log_dir)
    if args.pretrained != '':
        if '/' in args.pretrained:
            pt_exp = args.pretrained.split('/')[0]
        else:
            pt_exp = args.pretrained
        with open(f"{args.logdir}/{pt_exp}/config.json", 'r') as file:
            pt_args = json.load(file)
        pt_args_overwrite = {k: pt_args[k] for k in pt_overwrite_keys if k in pt_args}
        cfg_extra = {**cfg_extra, **pt_args_overwrite}
    cfg = TrainingConfig.from_args_and_dist(args, dist_info, cfg_extra)
    if (dist_info.rank == 0): # master
        make_clean_dir(f"{args.logdir}/{args.exp_name}")
        d = cfg.as_dict()
        with open(f"{args.logdir}/{args.exp_name}/config.json", 'w') as f:
            json.dump(d, f, indent=4)
            f.close()
    return cfg
