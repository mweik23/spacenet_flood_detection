from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from pathlib import Path
import torch.nn.functional as F


class TiledImageDataset(Dataset):
    """
    Expects a list of image filepaths. Yields random tiles of size tile_size.
    Optionally returns (image, label) if you have labels per image.
    """
    def __init__(self, files, tile_size=512, transforms=None, max_tiles_per_image=4, labels=None):
        self.files = files
        self.tile = tile_size
        self.tfms = transforms
        self.max_tiles = max_tiles_per_image
        self.labels = labels  # e.g., dict[path] = class_idx or anything you need

        #TODO:get shapes from df instead of opening images here
        # Preload sizes to avoid opening in __getitem__
        self.shapes = []
        for f in self.files:
            with Image.open(f) as im:
                self.shapes.append(im.size)  # (W, H)

        # Virtual length: each image contributes several tiles per epoch
        self._len = len(self.files) * self.max_tiles

    def __len__(self): return self._len

    def __getitem__(self, i):
        img_idx = i // self.max_tiles
        path = self.files[img_idx]
        W, H = self.shapes[img_idx]
        t = self.tile

        #TODO use tile sampler to get these coordinates
        # Sample a valid top-left for tile (if smaller than tile, will pad later)
        x0 = 0 if W <= t else random.randint(0, W - t)
        y0 = 0 if H <= t else random.randint(0, H - t)

        with Image.open(path) as im:
            im = im.convert("RGB")
            # Crop or pad to tile
            if W >= t and H >= t:
                im = im.crop((x0, y0, x0 + t, y0 + t))
            else:
                # pad small side(s) with reflection to keep content stats similar
                canvas = Image.new("RGB", (t, t))
                canvas.paste(im, (0, 0))
                im = canvas

        if self.tfms:
            im = self.tfms(im)

        y = None if self.labels is None else self.labels[path]
        return (im, y) if y is not None else im

# usage
from glob import glob
train_files = sorted(glob("data/train/**/*.jpg", recursive=True))
train_tiles = TiledImageDataset(train_files, tile_size=512, transforms=train_tfms, max_tiles_per_image=4)
train_loader = DataLoader(train_tiles, batch_size=32, shuffle=True, num_workers=num_workers,
                          pin_memory=True, prefetch_factor=2, persistent_workers=True, drop_last=True)

from typing import List, Tuple, Optional
import random
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

class ImageWithLabelList(Dataset):
    """Return lightweight metadata so we can open in collate_fn."""
    def __init__(self, img_paths: List[str], lab_paths: List[str], y_scalar: Optional[List[int]]=None):
        assert len(img_paths) == len(lab_paths)
        self.img_paths = img_paths
        self.lab_paths = lab_paths
        self.y_scalar = y_scalar

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, i):
        # Return paths and any per-image scalar target
        return {'pre-event image': self.img_paths[i], 
                'post-event image 1': None, 
                'post-event image 2': None,
                'label': self.lab_paths[i], 
                'scalar': (None if self.y_scalar is None else self.y_scalar[i])}

def _grid_boxes(W, H, tile, stride=None):
    tw, th = (tile, tile) if isinstance(tile, int) else tile
    sw, sh = (tw, th) if stride is None else ((stride, stride) if isinstance(stride, int) else stride)
    xs = list(range(0, max(1, W - tw + 1), sw))
    ys = list(range(0, max(1, H - th + 1), sh))
    if xs[-1] != W - tw: xs.append(max(0, W - tw))
    if ys[-1] != H - th: ys.append(max(0, H - th))
    return [(x, y, x + tw, y + th) for y in ys for x in xs]

def _pil_to_tensor_rgb(pil: Image.Image) -> torch.Tensor:
    # [C,H,W] float32 in [0,1]
    arr = np.asarray(pil, dtype=np.uint8).transpose(2,0,1)  # H,W,3 -> 3,H,W
    
    t = torch.from_numpy(arr).float() / 255.0
    return t

def _label_to_tensor(label_img_or_array, crop_box) -> torch.Tensor:
    # Accept either a single-channel grayscale PIL, an RGB-like multichannel PIL,
    # or a saved numpy .npy loaded earlier. Convert to [K,H,W] float32.
    x0, y0, x1, y1 = crop_box
    if isinstance(label_img_or_array, Image.Image):
        # If labels are stored as grayscale (soft probs in [0,255] or [0,1])
        # do not .convert("RGB"); keep mode
        lab = label_img_or_array.crop((x0, y0, x1, y1))
        arr = np.asarray(lab)
        if arr.ndim == 2:               # H,W
            arr = arr[None, ...]        # 1,H,W
        else:                           # H,W,K  -> K,H,W
            arr = np.moveaxis(arr, -1, 0)
        # scale if necessary (assume uint8 means probs * 255)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        return torch.from_numpy(arr)
    else:
        # If you loaded a numpy array earlier (K,H,W) or (H,W) and want to crop via slicing:
        arr = label_img_or_array
        if arr.ndim == 2:
            arr = arr[None, ...]
        x0, y0, x1, y1 = map(int, (x0,y0,x1,y1))
        cropped = arr[:, y0:y1, x0:x1].astype(np.float32)
        return torch.from_numpy(cropped)

def make_tile_collate(n_tiles: int,
                      tile_size: int | Tuple[int,int],
                      halo: int = 0,
                      stride: Optional[int | Tuple[int,int]] = None,
                      random_pick: bool = False,
                      img_transform=None,
                      # If labels need a transform that must mirror the image (e.g., resize),
                      # do it deterministically using the same parameters you used for the image.
                      ):
    def collate(batch: List[dict]):
        x_tiles, y_tiles, scalars = [], [], []
        for item in batch:
            img_path = item['pre-event image']
            lab_path = item['label']
            y_scalar = item['scalar']
            # Open image once
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size

                # Open label once; support PNG/TIFF labels or .npy (load externally if you prefer)
                if lab_path.endswith(".npy"):
                    lab = np.load(lab_path)  # (K,H,W) or (H,W)
                else:
                    lab = Image.open(lab_path)  # keep original mode (e.g., "F", "L", "I")

                corners = sample_corners() #TODO implement different sampling strategies

                for corner in corners:
                    full_box = (*[c-halo for c in corner], *[c+tile_size+2*halo for c in corner]) #(x0,y0,x1,y1)
                    # crop image
                    crop_box = (max(full_box[0], 0), max(full_box[1], 0), min(full_box[2], W), min(full_box[3], H))
                    tile_im = im.crop(crop_box)

                    if img_transform:
                        tile_im = img_transform(tile_im)  # must not change geometry vs label unless mirrored
                        if isinstance(tile_im, torch.Tensor):
                            x_tensor = tile_im
                        else:
                            # if transform returns PIL, convert
                            x_tensor = _pil_to_tensor_rgb(tile_im)
                    else:
                        x_tensor = _pil_to_tensor_rgb(tile_im)

                    # crop label with the SAME box
                    y_tensor = _label_to_tensor(lab if isinstance(lab, np.ndarray) else lab, crop_box)

                    pad_l = max(0, -full_box[0])
                    pad_t = max(0, -full_box[1])
                    pad_r = max(0, full_box[2] - W)
                    pad_b = max(0, full_box[3] - H)
                    
                    x_tensor = F.pad(x_tensor, (pad_l, pad_r, pad_t, pad_b), mode='reflect')
                    y_tensor = F.pad(y_tensor, (pad_l, pad_r, pad_t, pad_b), mode='reflect')
                    # (optional) geometry-preserving label postprocessing here
                    
                    x_tiles.append(x_tensor)
                    y_tiles.append(y_tensor)
                    scalars.append(y_scalar)

                if not isinstance(lab, np.ndarray) and hasattr(lab, "close"):
                    lab.close()

        X = default_collate(x_tiles)      # [B*n_tiles, C, h, w]
        Y = default_collate(y_tiles)      # [B*n_tiles, K, h, w] or [B*n_tiles, 1, h, w]
        S = None if all(v is None for v in scalars) else default_collate(scalars)
        return {'pre-event image': X, 'label': Y, 'scalar': S}
    return collate
