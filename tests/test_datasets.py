import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
import random, numpy as np
import os
from pathlib import Path
import pandas as pd
SRC_PATH = Path(__file__).parents[1] / 'src' / 'spacenet'
DATA_DIR = Path(__file__).parents[1] / 'data' / 'processed'
import sys
sys.path.append(str(SRC_PATH))
from dataset.data_processing import get_coords
from dataset.datasets import PathsDataset
from dataset.collate import TileCollator
from collections import deque
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test_loader_paths', action='store_true', help='Whether to run the loader paths test')
parser.add_argument('--test_collate_fn', action='store_true', help='Whether to run the collate fn test')
parser.add_argument('--test_dataloader', action='store_true', help='Whether to run the dataloader test')
parser.add_argument('--test_dataloader_real', action='store_true', help='Whether to run the dataloader real test')
args = parser.parse_args()

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def worker_init_fn(_wid: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    
def test_loader_paths(dataset, sampler, paths, batch_size, epoch=0):
    # Ensures all ranks advance the SAME permutation in lockstep
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,        # do NOT also set shuffle=True
        num_workers=1,
        worker_init_fn=worker_init_fn,
        collate_fn=default_collate,
        persistent_workers=True,
        pin_memory=True
    )
    sampler.set_epoch(epoch)
    batch = next(iter(loader))
    assert type(batch) is dict, f"Expected dict batch, got {type(batch)}"
    expected_keys = paths[0].keys()
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

def test_collate_fn(paths, batch_size=-1, img_size=1300, core_size=512, halo_size=32, stride=256, num_tiles=4, n_labels=1):
    collate = TileCollator(img_size=img_size, 
                           core_size=core_size, 
                           halo_size=halo_size, 
                           stride=stride, 
                           num_tiles=num_tiles, 
                           random_order=True, 
                           num_sets=10, 
                           verbose=True)
    if batch_size == -1:
        batch_size = len(paths)
        batch = paths
    else:
        batch = paths[:batch_size]
        
    collated = collate(batch)
    run_collated_batch_checks(collated, batch_size=batch_size, core_size=core_size, halo_size=halo_size, num_tiles=num_tiles, n_labels=n_labels)
    

def test_dataloaer(dataset, sampler, batch_size, img_size=1300, core_size=512, halo_size=32, stride=256, num_tiles=4, n_labels=1, epoch=0):
    collate = TileCollator(img_size=img_size,
                           core_size=core_size,
                           halo_size=halo_size,
                           stride=stride,
                           num_tiles=num_tiles,
                           random_order=True,
                           num_sets=10,
                           verbose=True)
    assert len(collate.tile_cache) == num_tiles, f"Expected {num_tiles} tiles in cache after initialization, got {len(collate.tile_cache)}"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,        # do NOT also set shuffle=True
        num_workers=1,
        worker_init_fn=worker_init_fn,
        collate_fn=collate,
        persistent_workers=False,
        pin_memory=True
    )
    sampler.set_epoch(epoch)
    collated = next(iter(loader))
    run_collated_batch_checks(collated, batch_size=batch_size, core_size=core_size, halo_size=halo_size, num_tiles=num_tiles, n_labels=n_labels)


def test_dataloader_real(repeated_dataset, sampler, num_workers, batch_size, num_sets, thresh=3, img_size=1300, core_size=512, halo_size=32, stride=256, num_tiles=4, n_labels=1, epoch=0):
    collate = TileCollator(img_size=img_size,
                           core_size=core_size,
                           halo_size=halo_size,
                           stride=stride,
                           num_tiles=num_tiles,
                           random_order=False,
                           num_sets=num_sets,
                           verbose=False)
    assert len(collate.sets) == num_sets, f"Expected {num_sets} sets in collate, got {len(collate.sets)}"
    assert len(collate.sets[0]) == num_tiles, f"Expected {num_tiles} tiles in each set, got {len(collate.sets[0])}"
    loader = DataLoader(
        repeated_dataset,
        batch_size=batch_size,
        sampler=sampler,        # do NOT also set shuffle=True
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collate,
        persistent_workers=False,
        pin_memory=True
    )
    sampler.set_epoch(epoch)
    batches = deque(maxlen=thresh)
    for i, batch in enumerate(loader):
        print(f"Processing batch {i}")
        batches.append(batch['pre-event image'])
        if len(batches) == thresh:
            if num_sets==1:
                print('checking equality of batches"')
                assert all(torch.equal(batches[0], batches[i]) for i in range(1, len(batches))), "with num_sets=1, batches should be identical but they are not"
                break
            else:
                assert not all(torch.equal(batches[0], batches[i]) for i in range(1, len(batches))), f"With num_sets={num_sets}, the probability of {thresh} identical batches is {(1/num_sets)**((thresh-1)*batch_size):.6f} but that is what happened. If you think this is a fluke, try increasing num_sets or thresh."
                break



if __name__ == '__main__':
    # ---- init DDP, get rank/world_size ----
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    seed = 42
    batch_size = 2

    # 1) Per-rank base seed for model/ops/augs (OK to differ by rank)
    base_op_seed = seed + rank
    set_global_seed(base_op_seed)

    # 2) Shared, fixed seed for the SAMPLER across all ranks
    sampler_seed = seed  # <-- SAME on all ranks

    splits_path = DATA_DIR / 'metadata' / 'splits.csv'
    splits_df = pd.read_csv(splits_path)

    image_names = splits_df[splits_df['split']=='train']['image_name']

    pre_image_dir = DATA_DIR / 'train' / 'PRE-event'
    label_dir = DATA_DIR / 'train' / 'labels'

    paths = [{'id': name,
            'pre-event image': str(pre_image_dir / f"{name}.png"),
            'labels': str(label_dir / f"labels_{get_coords(name)}.npy")} for name in image_names]

    dataset = PathsDataset(paths)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=sampler_seed,  # same seed everywhere
        drop_last=False,    # see note below
    )


    if args.test_loader_paths:
        test_loader_paths(dataset, sampler, paths, batch_size, epoch=0)
        print('loader_paths test passed!')
    if  args.test_collate_fn:
        test_collate_fn(paths)
        print('collate_fn test passed!')
    if args.test_dataloader:
        test_dataloaer(dataset, sampler, batch_size, epoch=0)
        print('dataloader test passed!')
    if args.test_dataloader_real:
        #create dataset of repeated paths for refresh test
        repeated_paths = [paths[0] for _ in range(10)]
        repeated_dataset = PathsDataset(repeated_paths)
        repeated_sampler = DistributedSampler(
            repeated_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=sampler_seed,  # same seed everywhere
            drop_last=False,    # see note below
        )
        test_dataloader_real(repeated_dataset, repeated_sampler, num_workers=2, batch_size=2, num_sets=1, thresh=3)
        print('collate randomness test passed with num_sets=1 (no randomness)!')
        test_dataloader_real(repeated_dataset, repeated_sampler, num_workers=2, batch_size=2, num_sets=100, thresh=3)
        print('collate randomness test passed with num_sets>0!')