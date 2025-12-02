from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
from shapely import wkt
from shapely.geometry import LineString, Polygon
import pandas as pd
import numpy as np
import cv2
import random
import tifffile as tiff
import json
from spacenet.dataset.data_utils import convert_to_image, resize, shuffle_dict_lists, get_coords
PROJECT_ROOT = Path(__file__).parents[3].resolve()
# add src to path

DATA_DIR = Path(PROJECT_ROOT / "data")
PROC_DIR = Path(DATA_DIR / "processed")

import random

def point_to_segment_distance(px, py, x0, y0, x1, y1):
    # vectorized distance from grid points (px,py) to segment (x0,y0)-(x1,y1)
    vx, vy = x1 - x0, y1 - y0
    wx, wy = px - x0, py - y0
    vv = vx*vx + vy*vy 
    t = (wx*vx + wy*vy) / vv
    t = np.clip(t, 0.0, 1.0)
    projx, projy = x0 + t*vx, y0 + t*vy
    dx, dy = px - projx, py - projy
    return np.sqrt(dx*dx + dy*dy), t  # return t if you want d_parallel too

def rasterize_lines(shape, segments, sigma=5, pad=200): 
    H, W = shape
    if len(segments)==0:
        return np.zeros(shape)
    d_min = np.full(shape, np.inf) # baseline inverse distance map, all inf
    num_segments = len(segments)
    for i, segment in enumerate(segments):
        #print(f"Processing segment {i+1}/{num_segments}")
        (x0,y0),(x1,y1) = segment.coords
        xmin = int(max(0, np.floor(min(x0,x1) - pad)))
        xmax = int(min(W-1, np.ceil (max(x0,x1) + pad)))
        ymin = int(max(0, np.floor(min(y0,y1) - pad)))
        ymax = int(min(H-1, np.ceil (max(y0,y1) + pad)))
        if xmin>xmax or ymin>ymax: 
            continue

        xs = np.arange(xmin, xmax+1)
        ys = np.arange(ymin, ymax+1)
        px, py = np.meshgrid(xs, ys)

        d, _ = point_to_segment_distance(px+0.5, py+0.5, x0, y0, x1, y1)
        d_min[ymin:ymax+1, xmin:xmax+1] = np.minimum(d_min[ymin:ymax+1, xmin:xmax+1], d)

    #return gaussian of the min distance
    return np.exp(-(d_min**2) / (2 * sigma**2))

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

def get_shapes(references, ImageId):
    """
    Get geometries for a specific image.
    """
    image_mask = references['ImageId'] == ImageId
    geoms = references[image_mask]['Wkt_Pix'].apply(wkt.loads)
    lines = geoms[geoms.apply(lambda x: x.geom_type == 'LineString')]
    polys = geoms[geoms.apply(lambda x: x.geom_type == 'Polygon')]
    if len(lines)==1:
        if lines.iloc[0] == LineString():
            lines = []
    if len(polys)==1:
        if polys.iloc[0] == Polygon():
            polys = []
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

class ImageProcessor:
    def __init__(self,
                regions,
                objects,
                label_channels=(),
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
        self.objects = objects
        self.label_channels = list(label_channels)
        self.dir_dict = dir_dict
        self.image_types = list(image_types)
        self.image_size_log = {'region_id': [], 'image_type': [], 'height': [], 'width': []}
        self.seed = seed
        self.processed_dir = processed_dir
        self.metadata_path = metadata_path
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.split_info = {
            "image_name": [],
            "split": [],
            "roads_flag": [],
            "buildings_flag": []
        }

    def convert_to_png(self, image_name, image_type, compress_level=1, ref_width=None, event_dir=None):
        #first open the pre-event image
        tif_path = self.image_paths[image_name][image_type]
        if tif_path is None:
            print(f"Image {image_name} of type {image_type} not found, skipping conversion.")
            return None, None
        img = tiff.imread(tif_path)
        img = convert_to_image(img)
        #create output directory if it doesn't exist
        event_dir.mkdir(parents=True, exist_ok=True)
        out_path = event_dir / (tif_path.stem + ".png")
        
        if ref_width is not None:
            print(f"Resizing image {image_name} of type {image_type}")
            img, new_size = resize(img, ref_width=ref_width)
        else:
            new_size = img.shape
        self.image_size_log['region_id'].append(image_name)
        self.image_size_log['image_type'].append(image_type)
        self.image_size_log['height'].append(new_size[0])
        self.image_size_log['width'].append(new_size[1])

        #convert to png and save
        Image.fromarray(img).save(out_path, compress_level=compress_level)
        return img, out_path

    def build_labels(self, image_name, img_shape, out_dir=None, sigma=5):
        label_idxs = {'roads': 0, 'buildings': 1}
        print(f"Building labels for image {image_name}")
        shapes = get_shapes(self.objects, image_name)
        shapes = {channel: shapes[label_idxs[channel]] for channel in self.label_channels if channel in label_idxs}
        for object_type, object_list in shapes.items():
            self.split_info[object_type + '_flag'].append(len(object_list)>0)
        labels={}
        labels_tup = tuple((rasterize_lines(img_shape, shapes['roads'], sigma=sigma)[None, :, :]
                      for ch in self.label_channels if ch == 'roads'))
        #---------- Add building rasterization to labels_tup expression above -----------
        if 'buildings' in self.label_channels:
            self.label_channels.remove('buildings')
            print("Building label rasterization not implemented yet.")
            self.label_channels = [ch for ch in self.label_channels if ch != 'buildings']
        #--------------------------------------------------------------------------------
        labels = np.concatenate(labels_tup, axis=0) if len(labels_tup)>0 else np.empty((0, img_shape[0], img_shape[1]))
        
        label_dir = out_dir / "labels" 
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / f"labels_{get_coords(image_name)}.npy"
        np.save(label_path, labels)
        return labels, label_path

    #image_names are the names event images such as
    def process_data(self, img_names=None, num_imgs=-1, val_size=0.2, compress_level=1, sigma=5):
        if img_names is None:
            img_names = list(self.image_paths.keys())
        if num_imgs !=-1:
            assert num_imgs>0, f"num_imgs must be positive or -1 but got {num_imgs}"
            assert num_imgs<=len(img_names), f"num_imgs {num_imgs} exceeds total number of images {len(img_names)}"
            img_names = random.sample(img_names, num_imgs)
            print(f"Processing images: {img_names}")
        train_names, val_names = train_test_split(img_names, test_size=val_size, random_state=self.seed)
        out_paths = {}
        for split, split_names in [("train", train_names), ("valid", val_names)]:
            out_dir = self.processed_dir / split
            out_dir.mkdir(parents=True, exist_ok=True)
            # TODO: I will use the vector data to build the soft label maps and save them as png files.
            for image_name in split_names:
                self.split_info['image_name'].append(image_name)
                self.split_info['split'].append(split)
                #prepare img types list
                img_types_copy = self.image_types.copy()
                img_types_copy.remove('pre-event image')
                #implement soft label map generation
                img_pre, out_path = self.convert_to_png(image_name, 'pre-event image', event_dir=out_dir / self.dir_dict['pre-event image'], compress_level=compress_level)
                out_paths[image_name] = {'pre-event image': out_path}
                if len(self.label_channels) > 0:
                    _, label_path = self.build_labels(image_name, img_pre.shape[:2], out_dir=out_dir, sigma=sigma)
                out_paths[image_name]['label'] = label_path
                #if there are post-event images, open them and align them with pre-event image
                for img_type in img_types_copy:
                    _, out_path = self.convert_to_png(image_name, img_type, ref_width=img_pre.shape[1], event_dir=out_dir / self.dir_dict[img_type], compress_level=compress_level)
                    out_paths[image_name][img_type] = out_path

        label_metadata = {'classes': self.label_channels,
                          'dtype': 'float32',
                          'layout': 'CHW',
                          'range': [0.0, 1.0],
                          'description': 'Soft label maps',
                          'sum_to_1': False
                          }
        with open(self.metadata_path / "label_metadata.json", 'w') as f:
            json.dump(label_metadata, f, indent=4)
        drop_keys = []
        for k, v in self.split_info.items():
            if len(v)==0:
                drop_keys.append(k)
        for k in drop_keys:
            self.split_info.pop(k)

        self.split_info = shuffle_dict_lists(self.split_info)
        
        split_info = pd.DataFrame(self.split_info)
        
        split_info.to_csv(self.metadata_path / "splits.csv", index=False)
        
        return out_paths


