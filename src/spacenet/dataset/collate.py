from typing import List, Tuple, Optional
import random
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
import numpy as np

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

class TileCollator:
    #assumes image and tiles are square and accepts only single integers for sizes
    def __init__(self, img_size, core_size, halo_size, stride, num_tiles):
        self.img_size = img_size
        self.core_size = core_size
        self.halo_size = halo_size
        self.stride = stride
        self.num_tiles = num_tiles
        self.refresh_grid()
        

    def sample_tile_coords(self):
        #sample anchor pixel
        px, py = np.random.randint(0, self.img_size-self.core_size, size=(2,))
        #check if pixel is within a stride of any edge
        top, bottom = py < self.stride, py >= self.img_size - self.core_size - self.stride
        left, right = px < self.stride, px >= self.img_size - self.core_size - self.stride
        #which tile does this pixel belong to?
        #tile num 0 is the tile belonging to the first full stride
        #tile num self.num_grid_(x,y)-1 is the tile belonging to the last fractional stride
        num_x = (px - self.ux) // self.stride
        num_y = (py - self.uy) // self.stride
        #if pixel belongs to an edge tile whose stride overlaps the next tile, sample between the two
        if left and num_x==0:
            num_x = np.random.randint(-1, 1)
        elif right and num_x == self.num_grid_x - 2:
            num_x = np.random.randint(self.num_grid_x - 2, self.num_grid_x)
        if top and num_y==0:
            num_y = np.random.randint(-1, 1)
        elif bottom and num_y == self.num_grid_y - 2:
            num_y = np.random.randint(self.num_grid_y - 2, self.num_grid_y)
        #check if tile was used already
        #calculate tile coordinates
        if num_x == -1:
            x0 = 0
        elif num_x == self.num_grid_x - 1:
            x0 = self.img_size - self.core_size
        else:
            x0 = num_x * self.stride + self.ux
        #same for y
        if num_y == -1:
            y0 = 0
        elif num_y == self.num_grid_y - 1:
            y0 = self.img_size - self.core_size
        else:
            y0 = num_y * self.stride + self.uy
        
        if len(self.overlapping_tiles) < self.max_tiles:
            if (x0, y0) in self.overlapping_tiles:
                return self.sample_tile_coords()  # recursively sample until unused tile is found
        else:
            print("no more non-overlapping tiles available, sampling unused tiles")
            if (x0, y0) in self.tile_cache:
                return self.sample_tile_coords()  # recursively sample until unused tile is found
        
        
        #store used tile coordinates
        self.tile_cache.append((x0, y0))
        new_overlapping_tiles = self.get_overlapping_tiles(x0, y0)
        self.overlapping_tiles.update(new_overlapping_tiles)
        return (x0, y0)
    
    def refresh_tiles(self):
        self.clear()
        for _ in range(self.num_tiles):
            self.sample_tile_coords()
            
    def refresh_grid(self):
        self.ux = np.random.randint(0, self.stride)
        self.uy = np.random.randint(0, self.stride)
        #number of gridlines for this arrangemnt
        self.num_grid_x = -(-(self.img_size - self.ux - self.core_size) // self.stride)
        self.num_grid_y = -(-(self.img_size - self.uy - self.core_size) // self.stride)
        self.overlapping_tiles = set()  # to keep track of overlapping tiles
        self.max_tiles = (self.num_grid_x+1) * (self.num_grid_y+1)
        self.tile_cache = []
        self.x_grid = np.clip(self.ux+self.stride*np.arange(-1, self.num_grid_x), 0, self.img_size - self.core_size)
        self.y_grid = np.clip(self.uy+self.stride*np.arange(-1, self.num_grid_y), 0, self.img_size - self.core_size)
        
    def get_overlapping_tiles(self, x0, y0):
        xs = self.x_grid[np.abs(self.x_grid - x0) < self.core_size]
        ys = self.y_grid[np.abs(self.y_grid - y0) < self.core_size]
        return [(x, y) for x in xs for y in ys]
    
    def clear(self):
        self.tile_cache = []
        self.overlapping_tiles = set()
    
    def __call__(self, batch: List[dict]):
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

                corners = random.sample(self.tile_cache, self.n_tiles)

                for corner in corners:
                    full_box = (*[c-self.halo for c in corner], *[c+self.core_size+2*self.halo for c in corner]) #(x0,y0,x1,y1)
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