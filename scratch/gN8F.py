from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

class PathsDataset(Dataset):
    """Return lightweight metadata so we can open in collate_fn."""
    def __init__(self, paths: List[dict[str]]):
        ks = [p.keys() for p in paths]
        assert all('id' in k for k in ks), "Some inputs do not have an id"
        assert all('pre-event image' in k for k in ks), "some inputs do not have a pre-event image"
        assert all('labels' in k for k in ks), "some inputs do not have a label"
        self.paths = paths

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        # Return paths and any per-image scalar target
        instance = self.paths[i]
        return {k: v for k, v in instance.items()}



