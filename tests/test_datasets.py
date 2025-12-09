import torch
import torch.distributed as dist
from torch.utils.data._utils.collate import default_collate
import os
from pathlib import Path
from dataclasses import asdict
import pytest
from spacenet.dataset.data_utils import get_im_size, get_num_classes
from spacenet.dataset.datasets import PathsDataset, get_dataloaders, get_paths
from spacenet.dataset.collate import TileCollator
from spacenet.configs import CollateConfig, DataConfig
from ml_tools.utils.random import set_global_seed
from ml_tools.utils.random import worker_init_base as worker_init_fn

from collections import deque

@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent

@pytest.fixture
def datadir(project_root):
    return project_root / 'data' / 'processed'

@pytest.fixture
def rank():
    return dist.get_rank() if dist.is_initialized() else 0

@pytest.fixture
def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

@pytest.fixture
def collate_cfg():
    return CollateConfig(
        core_size=512,
        halo_size=32,
        stride=256,
        num_tiles=4,
        num_sets=100
    )

@pytest.fixture
def make_data_cfg(collate_cfg, datadir):
    def _make_data_cfg(**overrides):
        kwargs = dict(
            datadir=str(datadir),
            batch_size=2,
            num_workers=0, #maybe parameterize
            num_data = -1,
            collate_cfg=collate_cfg,   # default
        )
        kwargs.update(overrides)
        return DataConfig(**kwargs)
    return _make_data_cfg

@pytest.fixture
def data_cfg(make_data_cfg):
    return make_data_cfg()

@pytest.fixture
def dataset(make_data_cfg):
    data_cfg = make_data_cfg(collate_cfg=None)
    return get_paths(Path(data_cfg.datadir), splits=('train',), num_data=data_cfg.num_data)

@pytest.fixture
def repeated_dataset(dataset):
    repeated_paths = [dataset['train'][0] for _ in range(10)]
    return {'train': PathsDataset(repeated_paths)}

def test_loader_paths(make_data_cfg, dataset, rank, world_size):
    seed = 42
    epoch = 0
    set_global_seed(seed, rank=rank)
    data_cfg = make_data_cfg(collate_cfg=None)
    batch_size = data_cfg.batch_size
    loader = get_dataloaders(
        datasets=dataset,
        rank=rank,
        world_size=world_size,
        collate_fn=default_collate,
        seed=seed,
        mode='pre-event only',
        **asdict(data_cfg))['train']
    
    loader.sampler.set_epoch(epoch)
    batch = next(iter(loader))
    assert type(batch) is dict, f"Expected dict batch, got {type(batch)}"
    
    expected_keys = dataset['train'][0].keys()
    assert all(k in batch for k in expected_keys), f"Missing keys in batch: {set(expected_keys) - set(batch.keys())}"
    assert all(k in expected_keys for k in batch.keys()), f"Unexpected keys in batch: {set(batch.keys()) - set(expected_keys)}"
    assert len(batch['id']) == batch_size, f"Expected batch size {batch_size}, got {len(batch['id'])}"
    assert all(isinstance(x, list) and len(x) == batch_size for x in batch.values()), "Each batch value should be a list of length batch_size"
    assert all(type(v[0]) is str for v in batch.values() if v), "All batch values should be strings but found non-string types"
    assert all(os.path.exists(p) for p in batch['pre-event image']), "All paths in batch should exist on disk but found 'pre-event image' paths that do not exist"
    assert (all(os.path.exists(p) for p in batch['labels'])), "All paths in batch should exist on disk but found 'labels' paths that do not exist"

def run_collated_batch_checks(collated, batch_size=1, core_size=512, halo_size=32, num_tiles=4, n_labels=1):
    assert 'pre-event image' in collated, "Collated batch missing 'pre-event image' key"
    assert 'labels' in collated, "Collated batch missing 'labels' key"
    expected_shape_data = torch.Size((batch_size*num_tiles, 3, core_size+2*halo_size, core_size+2*halo_size))
    expected_shape_labels = torch.Size((batch_size*num_tiles, n_labels, core_size+2*halo_size, core_size+2*halo_size))
    assert collated['pre-event image'].size() == expected_shape_data, f"Expected collated images shape {expected_shape_data}, got {collated['pre-event image'].size()}"
    assert collated['labels'].size() == expected_shape_labels, f"Expected collated labels shape {expected_shape_labels}, got {collated['labels'].size()}"
    assert torch.all(collated['pre-event image']>=0) and torch.all(collated['pre-event image']<=1), "Collated images should be normalized between 0 and 1"
    assert torch.all(collated['labels'] >= 0) and torch.all(collated['labels'] <= 1), "Collated labels should be normalized between 0 and 1"

def test_collate_fn(dataset, collate_cfg, datadir):
    batch_size = -1
    n_labels = get_num_classes(datadir)
    img_size, _ = get_im_size(Path(dataset['train'][0]['pre-event image']))
    collate = TileCollator(img_size=img_size, 
                           **asdict(collate_cfg))
    if batch_size == -1:
        batch_size = len(dataset['train'])
        batch = dataset['train']
    else:
        batch = dataset['train'][:batch_size]
        
    collated = collate(batch)
    run_collated_batch_checks(collated, 
                              batch_size=batch_size, 
                              core_size=collate_cfg.core_size, 
                              halo_size=collate_cfg.halo_size, 
                              num_tiles=collate_cfg.num_tiles, 
                              n_labels=n_labels)
    

def test_dataloader(dataset, data_cfg, rank, world_size):
    seed=42
    epoch = 0
    set_global_seed(seed, rank=rank)
    n_labels = get_num_classes(Path(data_cfg.datadir))
    img_size, _ = get_im_size(Path(dataset['train'][0]['pre-event image']))
    collate = TileCollator(img_size=img_size,
                           **asdict(data_cfg.collate_cfg))
    assert len(collate.tile_cache) == data_cfg.collate_cfg.num_tiles, f"Expected {data_cfg.collate_cfg.num_tiles} tiles in cache after initialization, got {len(collate.tile_cache)}"
    loader = get_dataloaders(
        datasets=dataset,
        rank=rank,
        world_size=world_size,
        collate_fn=collate,
        seed=seed,
        mode='pre-event only',
        **asdict(data_cfg))['train']
    loader.sampler.set_epoch(epoch)
    collated = next(iter(loader))
    run_collated_batch_checks(collated, 
                              batch_size=data_cfg.batch_size, 
                              core_size=data_cfg.collate_cfg.core_size, 
                              halo_size=data_cfg.collate_cfg.halo_size, 
                              num_tiles=data_cfg.collate_cfg.num_tiles, 
                              n_labels=n_labels)

@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("num_sets", [1, 100])
@pytest.mark.parametrize("random_order", [True, False])
def test_dataloader_real(data_cfg, 
                         repeated_dataset,
                         rank, 
                         world_size,
                         num_workers, 
                         num_sets, 
                         random_order):
    seed=42
    epoch = 0
    thresh = 3
    #override in collate_cfg
    data_cfg.collate_cfg.num_sets = num_sets
    data_cfg.collate_cfg.random_order = random_order
    #overwrite in data_cfg
    data_cfg.num_workers = num_workers
    set_global_seed(seed, rank=rank)
    
    img_size, _ = get_im_size(Path(repeated_dataset['train'][0]['pre-event image']))
    
    collate = TileCollator(img_size=img_size,
                           **asdict(data_cfg.collate_cfg))
    
    assert len(collate.sets) == num_sets, f"Expected {num_sets} sets in collate, got {len(collate.sets)}"
    assert len(collate.sets[0]) == data_cfg.collate_cfg.num_tiles, f"Expected {data_cfg.collate_cfg.num_tiles} tiles in each set, got {len(collate.sets[0])}"

    loader = get_dataloaders(
        datasets=repeated_dataset,
        rank=rank,
        world_size=world_size,
        collate_fn=collate,
        seed=seed,
        persistent_workers=False,
        pin_memory=True,
        mode='pre-event only',
        **asdict(data_cfg))['train']
    
    loader.sampler.set_epoch(epoch)
    batches = deque(maxlen=thresh)
    for i, batch in enumerate(loader):
        print(f"Processing batch {i}")
        batches.append(batch['pre-event image'])
        if len(batches) == thresh:
            if num_sets==1 and not random_order:
                print('checking equality of batches"')
                assert all(torch.equal(batches[0], batches[i]) for i in range(1, len(batches))), "with num_sets=1, batches should be identical but they are not"
                break
            else:
                assert not all(torch.equal(batches[0], batches[i]) for i in range(1, len(batches))), f"With num_sets={num_sets}, the probability of {thresh} identical batches is {(1/num_sets)**((thresh-1)*data_cfg.batch_size):.6f} but that is what happened. If you think this is a fluke, try increasing num_sets or thresh."
                break