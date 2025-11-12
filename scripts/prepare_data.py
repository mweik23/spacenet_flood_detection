from pathlib import Path
import json
import pandas as pd
import shutil

PROJECT_ROOT = Path(__file__).parents[1].resolve()
SRC_DIR = PROJECT_ROOT / "src" / "spacenet"
# add src to path
import sys
sys.path.append(str(SRC_DIR))
from dataset.data_processing import process_metadata, ImageProcessor

DATA_DIR = Path(PROJECT_ROOT / "data")
PROC_DIR = Path(DATA_DIR / "processed")

def main():
    num_imgs = 10
    if PROC_DIR.exists():
        shutil.rmtree(PROC_DIR)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    dfs = process_metadata(cities=('Germany', 'Louisiana-East'), 
                           raw_split='Training', 
                           rename_dict={'regions': 'label_image_mapping', 'objects': 'reference'},
                           metadata_path=PROC_DIR / "metadata",
                           df_types=('regions', 'objects')
                            )
    
    processor = ImageProcessor(regions=dfs['regions'],
                               objects=dfs['objects'],
                               label_channels=('roads',),
                               final_splits = ('train', 'valid'), 
                               image_types = ('pre-event image', 'post-event image 1', 'post-event image 2'),
                               metadata_path = PROC_DIR / "metadata"
                               )
    processor.process_data(num_imgs=num_imgs, val_size=0.2, compress_level=1, sigma=5)
    
    #save large translation log
    img_sz_log_df = pd.DataFrame(processor.image_size_log)
    img_sz_log_df.to_csv(PROC_DIR / "metadata" / "image_size_log.csv", index=False)
if __name__ == "__main__":
    main()