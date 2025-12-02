from spacenet.utils.cli import build_parser
from spacenet.configs import TrainerConfig, CollateConfig, DataConfig, ModelConfig, OptimConfig, SchedConfig, DistRuntime
from ml_tools.utils.distributed import setup_dist
from spacenet.utils.io import init_run
from pathlib import Path
from dataclasses import fields
from spacenet.dataset.data_utils import get_num_classes
import pytest

@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).parent.parent

@pytest.fixture
def arg_dict():
    return {
        'exp_name': "dummy",
        'batch_size': 2,
        'epochs': 2,
        'num_data': -1,
        'warmup_epochs': 5,
        'log_interval': 1,
        'patience': 3,
        'reduce_factor': 0.5
    }
@pytest.fixture
def args(arg_dict):
    argv = []
    for k, v in arg_dict.items():
        argv.append(f"--{k}")
        argv.append(str(v))
        
    parser = build_parser()
    return parser.parse_args(argv)

@pytest.fixture(scope="session")
def default_args():
    parser = build_parser()
    args = parser.parse_args([])
    #conver to dict
    return vars(args)

@pytest.fixture(scope="session")
def dist_info():
    return setup_dist()

@pytest.fixture(scope="session")
def dist_rt(dist_info):
    return DistRuntime.from_dist_info(dist_info)

@pytest.fixture
def full_cfg(args, dist_rt):
    return init_run(args, dist_rt)
    
def get_from_args(args_dict, attrs_names):
    out_dict = {}
    for name in attrs_names:
        if name in args_dict:
            out_dict[name] = args_dict[name]
    return out_dict

def update_args(new_args, base_args, attrs_names=None):
    updated = base_args.copy()
    if attrs_names is not None:
        for k, v in new_args.items():
            if k in attrs_names:
                updated[k] = v
    return updated

def get_args_for_class(ClassName, default_args, input_args):
    attrs = [f.name for f in fields(ClassName)]
    args = get_from_args(default_args, attrs)
    args = update_args(input_args, args, attrs)
    return args

def get_exp_data_cfg(default_args, input_args, project_root_val, **kwargs):
    collate_args = get_args_for_class(CollateConfig, default_args, input_args)
    data_args = get_args_for_class(DataConfig, default_args, input_args)
    data_args['collate_cfg'] = CollateConfig(**collate_args)
    data_args['datadir'] = str(project_root_val / data_args['datadir'])
    data_args.update(kwargs)
    return DataConfig(**data_args)

def get_exp_model_cfg(default_args, input_args, **kwargs):
    model_args = get_args_for_class(ModelConfig, default_args, input_args)
    model_args.update(kwargs)
    return ModelConfig(**model_args)

def test_data_config(full_cfg, default_args, arg_dict, dist_info, project_root):
    data_cfg = DataConfig.from_full(full_cfg)
    expected_num_workers = dist_info.num_workers
    expected_datacfg = get_exp_data_cfg(default_args, arg_dict, project_root, num_workers = expected_num_workers)
    assert data_cfg == expected_datacfg, f"DataConfig mismatch:\nGot: {data_cfg}\nExpected: {expected_datacfg}"

@pytest.mark.parametrize("in_channels", [1, 3, 4])
def test_model_config(full_cfg, default_args, arg_dict, in_channels):
    model_cfg = ModelConfig.from_full(full_cfg, in_channels=in_channels)
    exp_num_classes = get_num_classes(Path(full_cfg.datadir))
    expected_modelcfg = get_exp_model_cfg(default_args, arg_dict, in_channels=in_channels, num_classes=exp_num_classes)
    assert model_cfg == expected_modelcfg, f"ModelConfig mismatch:\nGot: {model_cfg}\nExpected: {expected_modelcfg}"
'''
def main(argv=None):
    
    
    trainer_cfg = TrainerConfig.from_full(full_cfg)
    
    optim_cfg   = OptimConfig.from_full(full_cfg, rename={'lr': 'peak_lr'})
    sched_config = SchedConfig.from_full(full_cfg, rename={'kind': 'sched_kind', 'mode': 'sched_mode'}) #TODO: ensure that these parameters exist in full config
    
    width = 80
    print("Trainer Config:", trainer_cfg)
    print(width*"-")
    print("Data Config:", data_cfg)
    print(width*"-")
    print("Model Config:", model_cfg)
    print(width*"-")
    print("Optimizer Config:", optim_cfg)
    print(width*"-")
    print("Scheduler Config:", sched_config)
    print(width*"-")
    print("Dist Config:", dist_rt.cfg)
    print(width*"-")
    print("Configuration successfully initialized.")
    
if __name__ == "__main__":
    in_channels = 3 
    
    
    
    
    
    
    
    main(argv)
'''