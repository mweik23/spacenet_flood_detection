from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from torch.utils.data import DataLoader, DistributedSampler
from spacenet.dataset.collate import TileCollator
from spacenet.dataset.data_utils import get_coords, get_im_size
from spacenet.configs import CollateConfig
import json


from ml_tools.utils.random import worker_init_base as worker_init

from pathlib import Path
import pandas as pd
from PIL import Image

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

def get_paths(data_dir: Path, splits=('train', 'valid'), num_data: int = -1) -> List[dict]:
    splits_path = data_dir / 'metadata' / 'splits.csv'
    splits_df = pd.read_csv(splits_path)
    paths = {}
    for split in splits:
        image_names = splits_df[splits_df['split']==split]['image_name']
        if num_data > 0:
            image_names = image_names[:num_data]

        pre_image_dir = data_dir / split / 'PRE-event'
        label_dir = data_dir / split / 'labels'

        paths[split] = PathsDataset([{'id': name,
                'pre-event image': str(pre_image_dir / f"{name}.png"),
                'labels': str(label_dir / f"labels_{get_coords(name)}.npy")} for name in image_names])
    return paths

def get_dataloaders(datasets: dict,
                    batch_size: int,
                    collate_fn: Optional[callable] = None,
                    collate_cfg: Optional[CollateConfig] = None,
                    num_workers: int = 0,
                    mode: str = 'pre-event only',
                    rank: int = 0,
                    world_size: int = 1,
                    seed: int = 42,
                    **kwargs
                    ) -> dict[str, DataLoader]:
    """Get train and valid dataloaders."""
    if collate_fn is None and collate_cfg is not None:
        if mode == 'pre-event only':
            img_size, _ = get_im_size(Path(datasets['train'][0]['pre-event image']))
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")
        collate_fn = TileCollator(img_size=img_size, **collate_cfg)
    if collate_fn is None:
        raise ValueError("Provide collate_fn or collate_cfg")
    loaders = {}
    if 'train' in datasets:
        train_sampler = DistributedSampler(datasets['train'], 
                                           shuffle=True, 
                                           seed=seed, 
                                           num_replicas=world_size, 
                                           rank=rank, 
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