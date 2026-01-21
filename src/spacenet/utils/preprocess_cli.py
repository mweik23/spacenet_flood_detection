import argparse
import json

def build_parser():
    parser = argparse.ArgumentParser(description='Spacenet Training Script')
    parser.add_argument('--data_dir', type=str, default='data/', metavar='N',
                        help='data directory from project root')
    parser.add_argument('--num_imgs', type=int, default=10, metavar='N',
                        help='number of images to process')
    parser.add_argument('--val_frac', type=float, default=0.2, metavar='N',
                        help='fraction of data to use for validation')
    parser.add_argument('--sigma_road', type=float, default=5.0, metavar='N',
                        help='sigma for road label smoothing')

    ############################################################                    
    parser.add_argument('--local_rank', type=int, default=0)
    
    return parser