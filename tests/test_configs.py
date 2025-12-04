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

@pytest.fixture
def optim_rename():
    return {'lr': 'peak_lr'}

@pytest.fixture
def sched_rename():
    return {'kind': 'sched_kind', 'mode': 'sched_mode'}
    
def get_from_args(args_dict, attrs_names, rename=None):
    if rename is None:
        rename = {}
    out_dict = {}
    for name in attrs_names:
        if rename.get(name, None) in args_dict.keys():
            out_dict[name] = args_dict[rename[name]]
        elif name in args_dict:
            out_dict[name] = args_dict[name]
    return out_dict
#TODO: implement rename
def update_args(new_args, base_args, attrs_names=None, rename=None):
    if rename is None:
        rename = {}
    updated = base_args.copy()
    for name in attrs_names:
        if rename.get(name, name) in new_args:
            updated[name] = new_args[rename.get(name, name)]
    return updated

def get_args_for_class(ClassName, default_args, input_args, rename=None):
    if rename is None:
        rename = {}
    #attribute list for the class
    attrs = [f.name for f in fields(ClassName)]
    print('type of default args:', type(default_args))
    args = get_from_args(default_args, attrs, rename=rename)
    args = update_args(input_args, args, attrs_names=attrs, rename=rename)
    return args

#---------------get expected configs-----------------

def get_exp_cfg(ClassName, default_args, input_args, rename=None, **kwargs):
    cfg_args = get_args_for_class(ClassName, default_args, input_args, rename=rename)
    cfg_args.update(kwargs)
    return ClassName(**cfg_args)

#---------------tests-----------------

def test_data_config(full_cfg, default_args, arg_dict, dist_info, project_root):
    data_cfg = DataConfig.from_full(full_cfg)
    collate_args = get_args_for_class(CollateConfig, default_args, arg_dict)
    expected_num_workers = dist_info.num_workers
    expected_data_cfg = get_exp_cfg(DataConfig, default_args, arg_dict, 
                                        num_workers = expected_num_workers, 
                                        collate_cfg = CollateConfig(**collate_args),
                                        datadir = str(project_root / arg_dict.get('datadir', default_args['datadir'])))
    assert data_cfg == expected_data_cfg, f"DataConfig mismatch:\nGot: {data_cfg}\nExpected: {expected_data_cfg}"
@pytest.mark.parametrize("in_channels", [1, 3, 4])
def test_model_config(full_cfg, default_args, arg_dict, in_channels):
    model_cfg = ModelConfig.from_full(full_cfg, in_channels=in_channels)
    exp_num_classes = get_num_classes(Path(full_cfg.datadir))
    expected_model_cfg = get_exp_cfg(ModelConfig, default_args, arg_dict, in_channels=in_channels, num_classes=exp_num_classes)
    assert model_cfg == expected_model_cfg, f"ModelConfig mismatch:\nGot: {model_cfg}\nExpected: {expected_model_cfg}"

def test_optim_config(full_cfg, default_args, arg_dict, optim_rename):
    optim_cfg = OptimConfig.from_full(full_cfg, rename=optim_rename)
    expected_optim_cfg = get_exp_cfg(OptimConfig, default_args, arg_dict, rename=optim_rename)
    assert optim_cfg == expected_optim_cfg, f"ModelConfig mismatch:\nGot: {optim_cfg}\nExpected: {expected_optim_cfg}"

def test_sched_config(full_cfg, default_args, arg_dict, sched_rename):
    sched_config = SchedConfig.from_full(full_cfg, rename=sched_rename)
    expected_sched_cfg = get_exp_cfg(SchedConfig, default_args, arg_dict, rename=sched_rename, lr_init_factor=full_cfg.start_lr/full_cfg.peak_lr)
    assert sched_config == expected_sched_cfg, f"SchedConfig mismatch:\nGot: {sched_config}\nExpected: {expected_sched_cfg}"

def test_trainer_config(full_cfg, default_args, arg_dict, project_root):
    trainer_cfg = TrainerConfig.from_full(full_cfg)
    raw_logdir = arg_dict.get('logdir', default_args['logdir'])
    exp_logdir = str(project_root / raw_logdir)
    expected_trainer_cfg = get_exp_cfg(TrainerConfig, default_args, arg_dict, logdir=exp_logdir)
    assert trainer_cfg == expected_trainer_cfg, f"TrainerConfig mismatch:\nGot: {trainer_cfg}\nExpected: {expected_trainer_cfg}"

