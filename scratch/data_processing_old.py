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
def resize(img, ref_size=(1300, 1300), crop='bottom', padding='top'):
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0))
    if ref_size[0] > img.shape[1]:
        interp = cv2.INTER_LANCZOS4
    else:
        interp = cv2.INTER_AREA
    new_size = (ref_size[0], int(img.shape[0] * ref_size[0] / img.shape[1])) # (W, H)
    scaled_img = cv2.resize(img, new_size, interpolation=interp)
    if img.shape[0] > img.shape[1]: # H > W
        print('cropping image from size: ', scaled_img.shape, 'to ref size: ', ref_size)
        if crop == 'bottom':
            padded_img = scaled_img[:ref_size[1], :]
        elif crop == 'top':
            padded_img = scaled_img[-ref_size[1]:, :]
    else:
        crop=None
        if img.shape[0] == img.shape[1]:
            padded_img = scaled_img
            print('no cropping or padding needed')
        else:
            print(f'padding {padding} of image from size: ', scaled_img.shape, 'to ref size: ', ref_size)
            if padding == 'top':
                padded_img = cv2.copyMakeBorder(scaled_img, max(0, ref_size[1] - new_size[1]), 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            elif padding == 'bottom':
                padded_img = cv2.copyMakeBorder(scaled_img, 0, max(0, ref_size[1] - new_size[1]), 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            else:
                print(f'no padding specified, returning scaled image of size: ', scaled_img.shape)
                padded_img = scaled_img
    return padded_img, crop
# Optional: use gradients to be illumination-invariant
# Optional: use gradients to be illumination-invariant
def grad_mag(img):
    img32 = img.astype(np.float32)
    gx = cv2.Scharr(img32, cv2.CV_32F, 1, 0)  # better than Sobel for small features
    gy = cv2.Scharr(img32, cv2.CV_32F, 0, 1)
    g  = cv2.magnitude(gx, gy)
    g  = (g - g.mean()) / (g.std() + 1e-6)
    return g

def get_warp_mat(input, ref, max_it=5000, conv_thresh=1e-6, use_grad=True, sigma=2):
    ref = cv2.GaussianBlur(ref, (0,0), sigma)
    input = cv2.GaussianBlur(input, (0,0), sigma)
    if use_grad:
        input = grad_mag(input)
        ref = grad_mag(ref)

    # Hanning window to reduce boundary artifacts
    win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    ref_w = ref * win
    input_w = input * win

    # Phase correlation returns (dx, dy) as float (subpixel)
    shift, response = cv2.phaseCorrelate(ref_w, input_w)
    dx, dy = shift  # shift to apply to 'mov' to align to 'ref'

    # Build translation matrix and warp
    warp_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    print(f"Estimated shift: dx={dx:.3f}, dy={dy:.3f}, peak response={response:.4f}")
    return warp_matrix


def get_warp_mat_old(input, ref, max_it=5000, conv_thresh=1e-6):
    # initial warp matrix for translation = 2x3 identity
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # define the stopping criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max_it,    # max iterations
        conv_thresh     # convergence threshold
    )

    # run ECC
    cc, warp_matrix = cv2.findTransformECC(
        ref,
        input,
        warp_matrix,
        motionType=cv2.MOTION_TRANSLATION,
        criteria=criteria
    )
    print("Correlation coefficient:", cc)
    print("Warp matrix:\n", warp_matrix)
    return warp_matrix

def align_w_pre(img_post, img_pre, crop='bottom', clahe=False, use_grad=False):
    ref_size = img_pre.shape[:-1]
    img_post, crop = resize(img_post, ref_size=ref_size, crop=crop, padding='top')

    print('resized image shape: ', img_post.shape)
    # convert to grayscale (required for ECC)
    gray_post = cv2.cvtColor(convert_to_image(img_post), cv2.COLOR_BGR2GRAY)
    gray_pre = cv2.cvtColor(convert_to_image(img_pre), cv2.COLOR_BGR2GRAY)
    
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(16, 16))
        gray_pre = clahe.apply(gray_pre)
        gray_post = clahe.apply(gray_post)
    # template = reference, input = to be warped
    template = gray_pre
    input_img = gray_post

    warp_matrix = get_warp_mat(input_img, template, use_grad=use_grad)
    
    #redo if I cropped the wrong side
    if round(warp_matrix[1, 2])>0 and crop=='bottom':
        crop = 'top'
        aligned, translation = align_w_pre(img_post, img_pre, crop=crop, clahe=clahe, use_grad=use_grad)
    if round(warp_matrix[1, 2])<0 and crop=='top':
        crop = 'bottom'
        aligned, translation = align_w_pre(img_post, img_pre, crop=crop, clahe=clahe, use_grad=use_grad)
    else:
        h, w = template.shape
        aligned = cv2.warpAffine(img_post, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        translation = warp_matrix[:, 2]
    return aligned, translation

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
        self.large_translation_log = {}
        self.seed = seed
        self.processed_dir = processed_dir
        self.metadata_path = metadata_path
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_jpg(self, image_id, image_type, quality=95, ref_img=None, threshold=20, event_dir=None):
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

        if ref_img is not None:
            print(f"Aligning image {image_id} of type {image_type} with pre-event image.")
            img_arr, translation = align_w_pre(img_arr, ref_img)
            if (np.abs(translation)>threshold).any():
                ref_path = self.processed_dir / self.dir_dict['pre-event image'] / (image_id + ".jpg")
                print(f"Warning: large translation for image {image_id} of type {image_type}: {translation}")
                self.large_translation_log[image_id] = {'image_type': image_type, 'translation': translation, 'jpg_path': jpg_path, 'ref_path': ref_path}

        #convert to jpg and save
        im = Image.fromarray(img_arr.astype(np.uint8))
        im.save(jpg_path, "JPEG", quality=quality, subsampling=0)
        return img_arr, jpg_path

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
                    _, jpg_path = self.convert_to_jpg(image_id, img_type, ref_img=img_pre, threshold=20, event_dir=out_dir / self.dir_dict[img_type], quality=quality)
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
