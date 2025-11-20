import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import os
from ml_tools.utils.random import set_global_seed, worker_init_base
from functools import partial


# Dummy dataset
class DummyDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, idx):
        return idx

def main():
    # --- On macOS, MUST use Gloo backend ---
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    os.environ["DL_RANK"] = str(rank)
    
    BASE_SEED = 1234

    # Per-rank seed
    rank_seed = set_global_seed(BASE_SEED, rank)
    print(f"[Rank {rank}] Global seed = {rank_seed}")

    # Dataset + sampler
    dataset = DummyDataset()

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=BASE_SEED,     # SAME on all ranks
    )
    worker_init = partial(worker_init_base, rank=rank, verbose=True)
    loader = DataLoader(
        dataset,
        batch_size=3,
        sampler=sampler,
        num_workers=2,        # will spawn 2 CPU workers per rank
        worker_init_fn=worker_init,
        persistent_workers=False,
    )

    sampler.set_epoch(0)

    for batch in loader:
        print(f"[Rank {rank}] got batch {batch}")


if __name__ == "__main__":
    main()