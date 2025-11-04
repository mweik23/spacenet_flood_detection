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
        default_collate=default_collate,
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

if __name__ == '__main__':
    # ---- init DDP, get rank/world_size ----
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    seed = 42
    batch_size = 1

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
            'labels': str(label_dir / f"labels_{get_coords(name)}.png")} for _, name in image_names.iterrows()]

    dataset = PathsDataset(paths)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=sampler_seed,  # same seed everywhere
        drop_last=False,    # see note below
    )


    test_loader_paths(dataset, sampler, paths, batch_size, epoch=0)
    print('loader_paths test passed!')