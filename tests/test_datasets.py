import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import random, numpy as np
import os
from pathlib import Path
import pandas as pd
SRC_PATH = Path(__file__).parents[1] / 'src' / 'spacenet'
DATA_DIR = Path(__file__).parents[1] / 'data' / 'processed'
import sys
sys.path.append(str(SRC_PATH))
from data_processing import get_coords
from dataset.datasets import PathsDataset


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def worker_init_fn(_wid: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

# ---- init DDP, get rank/world_size ----
rank = dist.get_rank() if dist.is_initialized() else 0
world_size = dist.get_world_size() if dist.is_initialized() else 1

seed = 42

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

loader = DataLoader(
    dataset,
    batch_size=bs,
    sampler=sampler,        # do NOT also set shuffle=True
    num_workers=4,
    worker_init_fn=worker_init_fn,
    persistent_workers=True,
    pin_memory=True,
)

for epoch in range(num_epochs):
    # Ensures all ranks advance the SAME permutation in lockstep
    sampler.set_epoch(epoch)

    for batch in loader:
        # training step ...
        pass