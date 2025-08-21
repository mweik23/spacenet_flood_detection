from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
from shapely import wkt
import pandas as pd
PROJECT_ROOT = Path(__file__).parents[1]
# add src to path
import sys
sys.path.append(str(PROJECT_ROOT / "src"))

DATA_DIR = Path("data/")
DST_DIR = Path("data/processed")

def process_data(ImageIds, image_types=['pre-event image'], val_size=0.3, seed=42, quality=95):
    train_ids, val_ids = train_test_split(ImageIds, test_size=val_size, random_state=seed)

    for split, split_ids in [("train", train_ids), ("valid", val_ids)]:
        out_dir = DST_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        # for each image id, I will make the neccessary transfromations of each image type 
        # and build the soft label maps. 
        # 
        # For pre and post event images, I will convert from tif to jpg
        # for post event images, I will resample to 1300 pixels in x direction and translate the image 
        # to align with pre-event image. Then I will save all images as jpg.
        #
        # I will use the vector data to build the soft label maps and save them as png files.
        for image_id in split_ids:
            img_types_copy = image_types.copy()
            #first open the pre-event image
            img_types_copy.remove('pre-event image')
            tif_path = image_paths[image_id]['pre-event image']
            img_pre = Image.open(tif_path).convert("RGB")
        
            #convert to jpg and save
            jpg_path = out_dir / (tif_path.stem + ".jpg")
            img.save(jpg_path, "JPEG", quality=quality, subsampling=0)

            #if there are post-event images, open them and align them with pre-event image
            for img_type in img_types_copy:
                #open the post-event image
                tif_path = image_paths[image_id][img_type]
                img = Image.open(tif_path).convert("RGB")
                #resample to 1300 pixels in x direction and align with pre-event image
                img = align_w_pre(img, img_pre) #TODO: bring in from jupyter notebook
                #convert to jpg and save
                jpg_path = out_dir / (tif_path.stem + ".jpg")
                img.save(jpg_path, "JPEG", quality=quality, subsampling=0)






    # optional: record split
    split_info = pd.DataFrame({
        "filename": [Id for Id in ImageIds],
        "split": ["train" if Id in train_files else "val" for Id in ImageIds]
    })
    split_info.to_csv(DST_DIR / "metadata" / "split.csv", index=False)

def get_raw_image_names(city, split='Training', image_types=['pre-event image']):
    df = pd.read_csv(DATA_DIR / f'{city}_{split}' / f"{city}_{split}_Public_label_image_mapping.csv")
    if 'pre-event image' not in image_types:
        image_types.append('pre-event image')
    return df[image_types]

def combine_dfs(cities, split='Training', name='label_image_mapping'):
    dfs = {}
    for city in cities:
        dfs[city] = pd.read_csv(DATA_DIR / f'{city}_{split}' / f"{city}_{split}_Public_{name}.csv")
        dfs[city]['city'] = city
    return pd.concat(dfs.values(), ignore_index=True)

def get_geoms(reference, ImageId):
    """
    Get geometries for a specific image.
    """
    image_mask = reference['ImageId'] == ImageId
    geoms = reference[image_mask]['Wkt_Pix'].apply(wkt.loads)
    lines = geoms[geoms.apply(lambda x: x.geom_type == 'LineString')]
    polys = geoms[geoms.apply(lambda x: x.geom_type == 'Polygon')]
    return lines, polys

if __name__ == "__main__":
    #---------input parameters---------
    cities = ['Germany', 'Louisiana-East']
    final_splits = ['train', 'valid']
    images_types = ['pre-event image']
    df_types = ['regions', 'objects']
    #-----------------------------------
    
    #some constants
    split_dict = {'train': 'Training', 'valid': 'Training', 'test': 'Test'}
    dir_dict = {'pre-event image': 'PRE-event', 'post-event image 1': 'POST-event', 'post-event image 2': 'POST-event'}
    df_names = {'regions': 'label_image_mapping', 'objects': 'reference'}

    #concatenate metadata and save to csv
    metadata_path = DST_DIR / "metadata"
    metadata_path.mkdir(parents=True, exist_ok=True)
    dfs = {}
    for df_type in df_types:
        dfs[df_type] = combine_dfs(cities, split='Training', name=df_names[df_type])
        #pd.to_csv(dfs[df_type], metadata_path / f"{df_type}.csv", index=False)

    image_paths = {dfs['regions'].iloc[idx]['pre-event image'].split('.tif')[0]: 
                   {img_type: DATA_DIR / f"{dfs['regions'].iloc[idx]['city']}_{split_dict[final_splits[0]]}" / dir_dict[img_type] / dfs['regions'].iloc[idx][img_type]
                        for img_type in images_types} for idx in dfs['regions'].index}
    
    #convert_and_split()