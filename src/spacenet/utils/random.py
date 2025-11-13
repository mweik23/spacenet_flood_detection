import torch
import numpy as np
import random
import os

def set_global_seed(seed: int, rank: int = 0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def worker_init_fn(_wid: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)