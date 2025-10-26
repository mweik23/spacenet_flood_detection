from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
from shapely import wkt
import pandas as pd
import numpy as np
import cv2
import random
PROJECT_ROOT = Path(__file__).parents[3].resolve()
# add src to path

DATA_DIR = Path(PROJECT_ROOT / "data")
PROC_DIR = Path(DATA_DIR / "processed")

def convert_to_image(img):
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0))
    return img

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

class ImageProcessor:
    def __init__(self,
                regions,
                final_splits = ('train', 'valid'), 
                image_types = ('pre-event image', 'post-event image 1', 'post-event image 2'), 
                dir_dict=None, 
                split_dict=None,
                seed=42, 
                processed_dir=PROC_DIR,
                metadata_path=PROC_DIR / "metadata"):
        if dir_dict is None:
            dir_dict = {'pre-event image': 'PRE-event', 'post-event image 1': 'POST-event', 'post-event image 2': 'POST-event'}
        if split_dict is None:
            split_dict = {'train': 'Training', 'valid': 'Training', 'test': 'Test'}
        self.image_paths = {row['pre-event image'].split('.tif')[0]:
                   {img_type: DATA_DIR / f"{row['city']}_{split_dict[final_splits[0]]}" / dir_dict[img_type] / row[img_type] if pd.notna(row[img_type]) else None
                        for img_type in image_types} for _, row in regions.iterrows()}
        self.dir_dict = dir_dict
        self.image_types = list(image_types)
        self.image_size_log = {'region_id': [], 'image_type': [], 'height': [], 'width': []}
        self.seed = seed
        self.processed_dir = processed_dir
        self.metadata_path = metadata_path
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_jpg(self, image_id, image_type, quality=95, ref_width=None, event_dir=None):
        #first open the pre-event image
        tif_path = self.image_paths[image_id][image_type]
        if tif_path is None:
            print(f"Image {image_id} of type {image_type} not found, skipping conversion.")
            return None, None
        with Image.open(tif_path) as img:
            img_arr = np.array(img.convert("RGB"))
        #create output directory if it doesn't exist
        event_dir.mkdir(parents=True, exist_ok=True)
        jpg_path = event_dir / (tif_path.stem + ".jpg")
        img = convert_to_image(img_arr)
        
        if ref_width is not None:
            print(f"Resizing image {image_id} of type {image_type}")
            img, new_size = resize(img, ref_width=ref_width)
        else:
            new_size = img.shape
        self.image_size_log['region_id'].append(image_id)
        self.image_size_log['image_type'].append(image_type)
        self.image_size_log['height'].append(new_size[0])
        self.image_size_log['width'].append(new_size[1])

        #convert to jpg and save
        im = Image.fromarray(img.astype(np.uint8))
        im.save(jpg_path, "JPEG", quality=quality, subsampling=0)
        return img, jpg_path

    def process_data(self, img_ids=None, num_imgs=-1, val_size=0.2, quality=95):
        if img_ids is None:
            img_ids = list(self.image_paths.keys())
        if num_imgs !=-1:
            assert num_imgs>0, f"num_imgs must be positive or -1 but got {num_imgs}"
            assert num_imgs<=len(img_ids), f"num_imgs {num_imgs} exceeds total number of images {len(img_ids)}"
            img_ids = random.sample(img_ids, num_imgs)
            print(f"Processing images: {img_ids}")
        train_ids, val_ids = train_test_split(img_ids, test_size=val_size, random_state=self.seed)
        out_paths = {}
        for split, split_ids in [("train", train_ids), ("valid", val_ids)]:
            out_dir = self.processed_dir / split
            out_dir.mkdir(parents=True, exist_ok=True)
            # TODO: I will use the vector data to build the soft label maps and save them as png files.
            for image_id in split_ids:
                #prepare img types list
                img_types_copy = self.image_types.copy()
                img_types_copy.remove('pre-event image')
                #TODO: decide whether to create a class for image processing
                img_pre, jpg_path = self.convert_to_jpg(image_id, 'pre-event image', event_dir=out_dir / self.dir_dict['pre-event image'], quality=quality)
                out_paths[image_id] = {'pre-event image': jpg_path}
                #if there are post-event images, open them and align them with pre-event image
                for img_type in img_types_copy:
                    _, jpg_path = self.convert_to_jpg(image_id, img_type, ref_width=img_pre.shape[1], event_dir=out_dir / self.dir_dict[img_type], quality=quality)
                    out_paths[image_id][img_type] = jpg_path

        # optional: record split
        split_info = pd.DataFrame({
            "image_id": [Id for Id in img_ids],
            "split": ["train" if Id in train_ids else "valid" for Id in img_ids]
        })
        split_info.to_csv(self.metadata_path / "splits.csv", index=False)
        
        return out_paths
def get_raw_image_names(city, split='Training', image_types=['pre-event image']):
    df = pd.read_csv(DATA_DIR / f'{city}_{split}' / f"{city}_{split}_Public_label_image_mapping.csv")
    if 'pre-event image' not in image_types:
        image_types.append('pre-event image')
    return df[image_types]

def combine_dfs(cities, split='Training', name='label_image_mapping', data_dir=DATA_DIR):
    dfs = {}
    for city in cities:
        dfs[city] = pd.read_csv(data_dir / f'{city}_{split}' / f"{city}_{split}_Public_{name}.csv")
        dfs[city]['city'] = city
    return pd.concat(dfs.values(), ignore_index=True)

def get_geoms(references, ImageId):
    """
    Get geometries for a specific image.
    """
    image_mask = references['ImageId'] == ImageId
    geoms = references[image_mask]['Wkt_Pix'].apply(wkt.loads)
    lines = geoms[geoms.apply(lambda x: x.geom_type == 'LineString')]
    polys = geoms[geoms.apply(lambda x: x.geom_type == 'Polygon')]
    return lines, polys

def process_metadata(cities=tuple(), raw_split='Training', rename_dict=None, df_types=('label_image_mapping', 'reference'), metadata_path=PROC_DIR):
    if rename_dict is None:
        df_types = ('label_image_mapping', 'reference')
        rename_dict = {k: k for k in df_types}
    #concatenate metadata and save to csv
    metadata_path.mkdir(parents=True, exist_ok=True)
    dfs = {}
    for df_type in df_types:
        dfs[df_type] = combine_dfs(cities, split=raw_split, name=rename_dict[df_type])
        dfs[df_type].to_csv(metadata_path / f"{df_type}.csv", index=False)
    return dfs

