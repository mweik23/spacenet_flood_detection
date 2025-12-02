import argparse
import json

def build_parser():
    parser = argparse.ArgumentParser(description='Spacenet Training Script')
    parser.add_argument('--exp_name', type=str, default='', metavar='N',
                        help='experiment_name')
    parser.add_argument('--test_mode', action='store_true', default=False,
                        help = 'test best model')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',   
                        help='input batch size for training')
    parser.add_argument('--num_data', type=int, default=-1, metavar='N',
                        help='number of samples')
    parser.add_argument('--epochs', type=int, default=35, metavar='N',
                        help='number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='number of warm-up epochs')             
    parser.add_argument('--seed', type=int, default=99, metavar='N',
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--val_interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before validation')
    parser.add_argument('--datadir', default='data/processed', metavar='N',
                        help='data directories')
    parser.add_argument('--logdir', type=str, default='logs', metavar='N',
                        help='folder to output logs')
    parser.add_argument('--peak_lr', type=float, default=1e-3, metavar='N',
                        help='learning rate')
    parser.add_argument('--num_workers', type=int, default=None, metavar='N',
                        help='number of workers for the dataloader')
    parser.add_argument('--patience', type=int, default=10, metavar='N',
                        help='learning rate scheduler')
    parser.add_argument('--threshold', type=float, default=1e-4, metavar='N',
                        help='threshold for lr scheduler to measure new optimum')
    parser.add_argument('--reduce_factor', type=float, default=0.1, metavar='N',
                        help='factor for LR scheduler if reduce')
    parser.add_argument('--start_lr', type=float, default=1e-4, metavar='N',
                        help='starting learning rate factor for warmup')
    parser.add_argument('--pretrained', type=str, default='', metavar='N',
                        help='directory with model to start the run with')
    parser.add_argument('--model_name', type=str, default='LorentzNet', metavar='N',
                        help='model name')
    parser.add_argument('--frozen_groups', type=json.loads, default='{}', metavar='N',
                        help='list of model groups to freeze')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='N',
                        help='momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='N',
                        help='weight decay for optimizer')
    parser.add_argument('--optimizer_type', type=str, default='AdamW', metavar='N',
                        help='type of optimizer to use')
    parser.add_argument('--sched_kind', type=str, default='warmup_plateau', metavar='N',
                        help='type of scheduler to use')
    parser.add_argument('--sched_mode', type=str, default='min', metavar='N',
                        help='mode of scheduler to use')
    parser.add_argument('--ld_optim_state', action='store_true', default=False,
                        help='load optimizer state from pretrained model')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='use automatic mixed precision training')
    parser.add_argument('--amp_dtype', type=str, default='bfloat16', metavar='N',
                        help='dtype for automatic mixed precision training')
    parser.add_argument('--mode', type=str, default='pre-event only', metavar='N',
                        help='training mode: pre-event only or pre- and post-event')
    parser.add_argument('--core_size', type=int, default=512, metavar='N',
                        help='core size for the model')
    parser.add_argument('--halo_size', type=int, default=32, metavar='N',
                        help='halo size for the model')
    parser.add_argument('--stride', type=int, default=256, metavar='N',
                        help='stride for the model')
    parser.add_argument('--num_tiles', type=int, default=4, metavar='N',
                        help='number of tiles for the model')
    parser.add_argument('--num_sets', type=int, default=100, metavar='N',
                        help='number of sets for the model')
    parser.add_argument('--base_channels', type=int, default=64, metavar='N',
                        help='number of base channels for the model')
    parser.add_argument('--depth', type=int, default=4, metavar='N',
                        help='depth of the model')
    ############################################################                    
    parser.add_argument('--local_rank', type=int, default=0)
    
    return parser