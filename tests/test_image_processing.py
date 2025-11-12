from pathlib import Path
import pandas as pd
import sys
from PIL import Image
import numpy as np
import shutil
import tifffile as tiff
SRC_DIR = Path(__file__).parents[1] / 'src' / 'spacenet'
sys.path.append(str(SRC_DIR))
from dataset.data_processing import ImageProcessor, process_metadata, get_shapes, convert_to_image

DATA_DIR = Path(__file__).parents[1].resolve() / "data"
PROC_DIR = Path(DATA_DIR / "processed")

def get_num_entries(name, cities, split):
    dfs = [pd.read_csv(DATA_DIR / f'{city}_{split}' / f"{city}_{split}_Public_{name}.csv") for city in cities]
    return sum(len(df) for df in dfs)

def coord2pix(x, L=1300):
    return int(np.clip(round(x-0.5), a_min=0, a_max=L-1))

def road_tests(label_array, objects_df, image_id, sigma=5):
    H, W = label_array.shape
    roads, _ = get_shapes(objects_df, image_id)
    #TODO: be more generous with the distance threshold
    max_sq_dist = 2*0.5**2
    min_label_val = np.exp(-max_sq_dist/(2*sigma**2))
    for road in roads:
        p0, p1 = road.coords
        x0, y0 = (coord2pix(x, L=L) for x, L in zip(p0, (H, W)))
        x1, y1 = ((coord2pix(x, L=L) for x, L in zip(p1, (H, W))))
        assert label_array[y0, x0] >= min_label_val, f"Label value {label_array[y0, x0]} at the road endpoint at {(y0, x0)} is less than expected minimum {min_label_val}"
        assert label_array[y1, x1] >= min_label_val, f"Label value {label_array[y1, x1]} at the road endpoint at {(y1, x1)} is less than expected minimum {min_label_val}"
    assert label_array.max() <= 1.0, f"Maximum label value {label_array.max()} exceeds 1.0"
    assert label_array.min() >= 0.0, f"Minimum label value {label_array.min()} is less than 0.0"
    mean_val = label_array.mean()
    if len(roads) > 0:
        assert mean_val > 0.0, f"Mean label value {mean_val} is not greater than 0.0"
    assert mean_val < 0.25, f"Mean label value {mean_val} is not less than 0.25 and therefore it is likely not sparse enough"
    print("✅ road label values test passed.")

def label_shape_test(labels, label_channels):
    label_channels = tuple((ch for ch in label_channels if ch!='buildings'))
    expected_label_shape = (len(label_channels), 1300, 1300)
    assert labels.ndim == 3 and labels.shape == expected_label_shape, f"Processed label shape {labels.shape} is not as expected {expected_label_shape}"
    print("✅ processed label shape test passed.")

def main():
    #clear processed directory
    if PROC_DIR.exists():
        shutil.rmtree(PROC_DIR)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df_types = ('regions', 'objects')
    rename_dict = {'regions': 'label_image_mapping', 'objects': 'reference'}
    num_imgs = 8
    label_channels = ('roads',)
    sigma = 5
    
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
                               objects=dfs['objects'],
                               label_channels=label_channels,
                               final_splits = ('train', 'valid'), 
                               image_types = ('pre-event image', 'post-event image 1', 'post-event image 2'),
                               metadata_path = PROC_DIR / "metadata"
                               )
    output_paths = processor.process_data(num_imgs=num_imgs, val_size=0.5, compress_level=1, sigma=sigma)
    assert len(output_paths) == num_imgs, f"Number of processed images {len(output_paths)} does not match expected {num_imgs}"
    image_id = list(output_paths.keys())[0]

    #check dimensions of pre-event image
    img = Image.open(output_paths[image_id]['pre-event image'])
    ar = np.array(img)
    #check pre-event image shape
    assert ar.ndim == 3 and ar.shape == (1300, 1300, 3), f"Processed image shape {ar.shape} is not as expected (1300, 1300, 3)"
    print("✅ processed pre-event image shape test passed.")
    
    #compare pre-event image to itself before conversion
    pre_event_path = processor.image_paths[image_id]['pre-event image']
    #open tif image
    ar_orig = tiff.imread(pre_event_path)
    ar_orig = convert_to_image(ar_orig)    
    #check that the pixel values are the same
    diff = ar.astype(np.int16) - ar_orig.astype(np.int16)
    assert np.all(diff==0), f"Max pixel difference between original and processed pre-event image is {np.abs(diff).max()}, expected 0"
    print("✅ processed pre-event image comparison test passed.")
    
    labels = np.load(output_paths[image_id]['label'])
    #check label shape
    label_shape_test(labels, label_channels)
    
    #compare label with original geojson
    if len(label_channels)==0:
        assert labels == np.empty((0, 1300, 1300)), "Labels should be empty array when no label channels are specified"
    else:
        for i, ch in enumerate(label_channels):
            if ch == 'roads':
                road_tests(labels[i, :, :], dfs['objects'], image_id, sigma=sigma)
            
    #check dimensions of a post-event image
    post_event_path = output_paths[image_id]['post-event image 1']
    img_post = Image.open(post_event_path)
    ar_post = np.array(img_post)
    #check post-event image shape
    assert ar_post.ndim == 3 and ar_post.shape[1] == 1300, f"Processed post-event image width {ar_post.shape[1]} is not as expected 1300"
    print("✅ processed post-event image shape test passed.")
    #testing image_size_log
    print("Image size log:", processor.image_size_log)
    img_sz_log_df = pd.DataFrame(processor.image_size_log)
    #save imgage size log for manual inspection
    img_sz_log_df.to_csv(PROC_DIR / "metadata" / "image_size_log.csv", index=False)
    assert len(img_sz_log_df['region_id'].unique()) == num_imgs, f"Number of entries in image_size_log {len(img_sz_log_df['region_id'].unique())} does not match expected {num_imgs}"
    assert (img_sz_log_df['width']==1300).all(), f"Not all images have expected width 1300"
    assert (img_sz_log_df[img_sz_log_df['image_type'] == 'pre-event image']['height'] == 1300).all(), f"Not all pre-event images have expected height 1300"
    print("✅ image_size_log test passed.")
    #-------------------------------------------------------------
    print("✅✅ All tests passed successfully. ✅✅")
if __name__ == "__main__":
    main()