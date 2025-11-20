from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from torch.utils.data import DataLoader, DistributedSampler
from ml_tools.utils.distributed import DistInfo
from spacenet.dataset.collate import TileCollator
from spacenet.dataset.data_processing import get_coords
from spacenet.utils.random import worker_init_base as worker_init
from pathlib import Path
import pandas as pd

def get_paths(data_dir: Path) -> List[dict]:
    splits_path = data_dir / 'metadata' / 'splits.csv'
    splits_df = pd.read_csv(splits_path)

    image_names = splits_df[splits_df['split']=='train']['image_name']

    pre_image_dir = data_dir / 'train' / 'PRE-event'
    label_dir = data_dir / 'train' / 'labels'

    paths = [{'id': name,
            'pre-event image': str(pre_image_dir / f"{name}.png"),
            'labels': str(label_dir / f"labels_{get_coords(name)}.npy")} for name in image_names]
    return paths

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

def get_dataloaders(datasets: dict,
                    batch_size: int,
                    collate_fn,
                    collate_cfg: Optional[dict] = None,
                    num_workers: int = 0,
                    dist_info: DistInfo = None,
                    seed: int = 42
                    ) -> dict[str, DataLoader]:
    """Get train and valid dataloaders."""
    if collate_fn is None and collate_cfg is not None:
        collate_fn = TileCollator(**collate_cfg)
    if collate_fn is None:
        raise ValueError("Provide collate_fn or collate_cfg")
    loaders = {}
    if 'train' in datasets:
        train_sampler = DistributedSampler(datasets['train'], 
                                           shuffle=True, 
                                           seed=seed, 
                                           num_replicas=dist_info.world_size, 
                                           rank=dist_info.rank, 
                                           drop_last=False)
    else:
        train_sampler = None
    for split, dataset in datasets.items():
        assert isinstance(dataset, Dataset), f"Dataset for split {split} is not a torch Dataset"
        train_sampler
        loaders[split] = DataLoader(
            datasets['split'],
            sampler=train_sampler if split=='train' else None,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=worker_init,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

    return loaders