from pathlib import Path
from typing import Tuple
from PIL import Image
import json
import random
import cv2

def get_channels(image_path: Path) -> int:
    # Assuming datasets is a dict with splits as keys and PathsDataset as values
    # Get the first image path from the 'train' split
    with Image.open(image_path) as img:
        return len(img.getbands())

def get_num_classes(datadir: Path) -> int:
    with open(datadir / 'metadata' / 'label_metadata.json', 'r') as f:
        label_metadata = json.load(f)
    return len(label_metadata['classes'])

def get_im_size(path: Path) -> Tuple[int, int]:
    """Get image size (H, W) without loading full image into memory."""
    with Image.open(path) as img:
        return img.size[1], img.size[0]  # PIL gives (W, H)
    
def shuffle_dict_lists(d):
    # Convert dict of lists → list of tuples
    keys = list(d.keys())
    rows = list(zip(*[d[k] for k in keys]))
    
    # Shuffle rows in place
    random.shuffle(rows)
    
    # Unzip back to synchronized lists
    shuffled = {k: [] for k in keys}
    for row in rows:
        for k, value in zip(keys, row):
            shuffled[k].append(value)
    
    return shuffled

def convert_to_image(img):
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0))
    return img

def get_coords(image_name):
    image_id_len = len(image_name.split('_')[0])
    return image_name[image_id_len+1:]

''' Resize function to maintain aspect ratio and use appropriate interpolation based on size
ref_size: tuple of (width, height) to resize the image to
'''
def resize(img, ref_width=1300):
    assert len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[-1] == 3), f"Invalid image shape: {img.shape}"
    if ref_width > img.shape[1]:
        interp = cv2.INTER_LANCZOS4
    else:
        interp = cv2.INTER_AREA
    new_size = (ref_width, int(img.shape[0] * ref_width / img.shape[1])) # (W, H)
    scaled_img = cv2.resize(img, new_size, interpolation=interp)
    return scaled_img, scaled_img.shape