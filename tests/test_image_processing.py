from pathlib import Path
import pandas as pd
import sys
from PIL import Image
import numpy as np
import shutil
SRC_DIR = Path(__file__).parents[1] / 'src' / 'spacenet'
sys.path.append(str(SRC_DIR))
from dataset.data_processing import ImageProcessor, process_metadata

DATA_DIR = Path(__file__).parents[1].resolve() / "data"
PROC_DIR = Path(DATA_DIR / "processed")

def get_num_entries(name, cities, split):
    dfs = [pd.read_csv(DATA_DIR / f'{city}_{split}' / f"{city}_{split}_Public_{name}.csv") for city in cities]
    return sum(len(df) for df in dfs)

def main():
    #clear processed directory
    if PROC_DIR.exists():
        shutil.rmtree(PROC_DIR)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df_types = ('regions', 'objects')
    rename_dict = {'regions': 'label_image_mapping', 'objects': 'reference'}
    num_imgs = 4
    
    #------------test process_metadata----------------------
    dfs = process_metadata(cities=('Germany', 'Louisiana-East'), 
                           raw_split='Training', 
                           rename_dict=rename_dict,
                           metadata_path=PROC_DIR / "metadata",
                           df_types=df_types
                            )

    assert all(k in dfs for k in df_types), f"keys of dfs are {dfs.keys()}, expected {df_types}"
    for df_type in df_types:
        num_entries = get_num_entries(rename_dict[df_type], cities=('Germany', 'Louisiana-East'), split='Training')
        assert len(dfs[df_type]) == num_entries, f"Number of entries in {df_type} dataframe {len(dfs[df_type])} does not match expected {num_entries}"
    print("✅ process_metadata tests passed.")
    #--------------------------------------------------------
    
    #------------test ImageProcessor.process_data----------------------
    processor = ImageProcessor(regions=dfs['regions'],
                               final_splits = ('train', 'valid'), 
                               image_types = ('pre-event image', 'post-event image 1', 'post-event image 2'),
                               metadata_path = PROC_DIR / "metadata"
                               )
    output_paths = processor.process_data(num_imgs=num_imgs, val_size=0.5, quality = 99)
    assert len(output_paths) == num_imgs, f"Number of processed images {len(output_paths)} does not match expected {num_imgs}"
    image_id = list(output_paths.keys())[0]

    #check dimensions of pre-event image
    with Image.open(output_paths[image_id]['pre-event image']) as im:
        im = im.convert("RGB")
        ar = np.array(im)
        #check pre-event image shape
        assert ar.ndim == 3 and ar.shape == (1300, 1300, 3), f"Processed image shape {ar.shape} is not as expected (1300, 1300, 3)"
    print("✅ processed pre-event image shape test passed.")
    #compare pre-event image to itself before conversion
    pre_event_path = processor.image_paths[image_id]['pre-event image']
    #open tif image
    with Image.open(pre_event_path) as im:
        im = im.convert("RGB")
        ar_orig = np.array(im)
        #check that the pixel values are similar (within 5 units)
        diff = np.abs(ar.astype(np.int16) - ar_orig.astype(np.int16))
        max_diff = diff.max()
        assert max_diff <= 5, f"Max pixel difference between original and processed pre-event image is {max_diff}, expected <=5"
    print("✅ processed pre-event image comparison test passed.")
    #check dimensions of a post-event image
    post_event_path = output_paths[image_id]['post-event image 1']
    with Image.open(post_event_path) as im:
        im = im.convert("RGB")
        ar_post = np.array(im)
        #check post-event image shape
        assert ar_post.ndim == 3 and ar_post.shape[1] == 1300, f"Processed post-event image width {ar_post.shape[1]} is not as expected 1300"
    print("✅ processed post-event image shape test passed.")
    #testing image_size_log
    print("Image size log:", processor.image_size_log)
    img_sz_log_df = pd.DataFrame(processor.image_size_log)
    assert len(img_sz_log_df['region_id'].unique()) == num_imgs, f"Number of entries in image_size_log {len(img_sz_log_df['region_id'].unique())} does not match expected {num_imgs}"
    assert (img_sz_log_df['width']==1300).all(), f"Not all images have expected width 1300"
    assert (img_sz_log_df[img_sz_log_df['image_type'] == 'pre-event image']['height'] == 1300).all(), f"Not all pre-event images have expected height 1300"
    print("✅ image_size_log test passed.")
    #-------------------------------------------------------------
    print("✅✅ All tests passed successfully. ✅✅")
if __name__ == "__main__":
    main()