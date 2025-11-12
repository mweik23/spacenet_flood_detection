from shared_lib.utils.distributed import setup_dist, DistInfo
import torch
from spacenet.dataset.datasets import get_dataloaders
from spacenet.utils.cli import build_parser
def main(argv=None):
    #get arguments
    parser = build_parser()
    args = parser.parse_args(argv)
    #set up distributed training
    dist_info: DistInfo = setup_dist(arg_num_workers=args.num_workers)
    if dist_info.is_primary:
        print(dist_info)
    device = torch.device(dist_info.device_type)
    path_datasets = {}
    loaders = get_dataloaders(
        datasets=path_datasets,
        batch_size=args.batch_size,
        collate_fn=None,  # TODO: add collate function
        num_workers=args.num_workers,
        dist_info=dist_info,
    )