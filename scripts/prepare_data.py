from pathlib import Path
import json
import pandas as pd
import shutil
from spacenet.utils.preprocess_cli import build_parser

PROJECT_ROOT = Path(__file__).parents[1].resolve()
SRC_DIR = PROJECT_ROOT / "src" / "spacenet"
# add src to path
import sys
sys.path.append(str(SRC_DIR))
from dataset.data_processing import process_metadata, ImageProcessor

def main():
    parser = build_parser()
    args = parser.parse_args()
    data_dir = PROJECT_ROOT / args.data_dir
    proc_dir = data_dir / "processed"
    if proc_dir.exists():
        shutil.rmtree(proc_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    dfs = process_metadata(cities=('Germany', 'Louisiana-East'), 
                           raw_split='Training', 
                           rename_dict={'regions': 'label_image_mapping', 'objects': 'reference'},
                           metadata_path=proc_dir / "metadata",
                           df_types=('regions', 'objects')
                            )
    
    processor = ImageProcessor(regions=dfs['regions'],
                               objects=dfs['objects'],
                               label_channels=('roads',),
                               final_splits = ('train', 'valid'), 
                               image_types = ('pre-event image', 'post-event image 1', 'post-event image 2'),
                               metadata_path = proc_dir / "metadata"
                               )
    processor.process_data(num_imgs=args.num_imgs, val_size=args.val_frac, compress_level=1, sigma=args.sigma_road)
    
    #save large translation log
    img_sz_log_df = pd.DataFrame(processor.image_size_log)
    img_sz_log_df.to_csv(proc_dir / "metadata" / "image_size_log.csv", index=False)
if __name__ == "__main__":
    main()