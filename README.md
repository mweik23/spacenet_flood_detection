# SpaceNet Segmentation Model
A CNN-based machine learning model that performs segmentation on satellite images.

## Overview
The goal of this project is to build a model that identifies roads and buildings in satellite images from the SpaceNet dataset and classifies the objects as flooded or not flooded. The model uses a UNet architechture and a low bias tiling strategy for training. The model is trained on images of various regions of Germany and Louisiana-East and tested on Louisiana-West. Directions for obtaining all datasets ccan be found at [Kaggle](https://www.kaggle.com/code/virajkadam/spacenet8-data-download-and-eda)


## Installation
### Prerequisites
- [python 3.9](https://www.python.org/downloads/release/python-390/)
- [pip](https://pip.pypa.io/en/stable/installation/)

### Local Development
#### CPU
```bash
conda env create -f environment.cpu.yml
conda activate spacenet
pip install -e ".[dev]"

```
#### CUDA
```bash
conda env create -f environment.cuda.yml
conda activate spacenet
pip install -e ".[dev]"
```

### Developing ml_tools alongside spacenet
If you are actively developing `ml_tools`, install it in editable mode
before installing `spacenet`:

```bash
pip uninstall ml_tools
pip install -e ../ml_tools
pip install -e .
```
## Command-Line Usage

### Preprocessing
#### Example Usage
```bash
python scripts/prepare_data.py --data_dir data/ --num_data 20 --val_frac 0.2 
```
The preprocessing script grabs data from `data_dir` which should contain one sub-directory for each location (Germany or Louisiana-East). The processed data is saved to a sub-directory called `processed`.

### Training
#### Example Usage
```bash
args=(
  --exp_name "test_run"
  --batch_size 2
  --num_data -1
  --epochs 3
  --warmup_epochs 5
  --peak_lr 1e-3
  --start_lr 5e-4
  --patience 3
  --threshold 1e-4
  --reduce_factor 0.3
  --seed 42
  --weight_decay 1e-4
  --mode "pre-event only"
  --core_size 512
  --halo_size 32
  --stride 256
  --num_tiles 4
  --num_sets 100
  --base_channels 16
  --depth 2
  --log_interval 1
  --freeze_bn
)

python scripts/train.py "${args[@]}"
```

#### Training Script Arguments

- **`--exp_name`** (default: `""`)  
  Experiment name.

- **`--test_mode`** (default: `False`)  
  Only perform testing epoch.

- **`--batch_size`** (default: `32`)  
  Input batch size for training.

- **`--num_data`** (default: `-1`)  
  Number of samples.

- **`--epochs`** (default: `35`)  
  Number of training epochs.

- **`--warmup_epochs`** (default: `5`)  
  Number of warm-up epochs.

- **`--seed`** (default: `99`)  
  Random seed.

- **`--test_seed`** (default: `None`)  
  Random seed for test mode.

- **`--log_interval`** (default: `100`)  
  How many batches to wait before logging training status.

- **`--val_interval`** (default: `1`)  
  How many epochs to wait before validation.

- **`--datadir`** (default: `"data/processed"`)  
  Data directories.

- **`--logdir`** (default: `"logs"`)  
  Folder to output logs.

- **`--peak_lr`** (default: `1e-3`)  
  Learning rate.

- **`--num_workers`** (default: `None`)  
  Number of workers for the dataloader.

- **`--patience`** (default: `10`)  
  Patience for learning rate scheduler.

- **`--threshold`** (default: `1e-4`)  
  Threshold for lr scheduler to measure new optimum.

- **`--reduce_factor`** (default: `0.1`)  
  Factor for LR scheduler if reduce.

- **`--start_lr`** (default: `1e-4`)  
  Starting learning rate factor for warmup.

- **`--pretrained`** (default: `""`)  
  Directory with model to start the run with.

- **`--model_name`** (default: `"UNet"`)  
  Model name.

- **`--momentum`** (default: `0.9`)  
  Momentum for SGD optimizer.

- **`--weight_decay`** (default: `1e-4`)  
  Weight decay for optimizer.

- **`--optimizer_type`** (default: `"AdamW"`)  
  Type of optimizer to use.

- **`--sched_kind`** (default: `"warmup_plateau"`)  
  Type of scheduler to use.

- **`--sched_mode`** (default: `"min"`)  
  Mode of scheduler to use.

- **`--ld_optim_state`** (default: `False`)  
  Load optimizer state from pretrained model.

- **`--use_amp`** (default: `False`)  
  Use automatic mixed precision training.

- **`--amp_dtype_str`** (default: `"bfloat16"`)  
  Dtype for automatic mixed precision training.

- **`--mode`** (default: `"pre-event only"`)  
  Training mode: pre-event only or pre- and post-event.

- **`--core_size`** (default: `512`)  
  Core size for the model.

- **`--halo_size`** (default: `32`)  
  Halo size for the model.

- **`--stride`** (default: `256`)  
  Stride for the model.

- **`--num_tiles`** (default: `4`)  
  Number of tiles for the model.

- **`--num_sets`** (default: `100`)  
  Number of sets for the model.

- **`--base_channels`** (default: `64`)  
  Number of base channels for the model.

- **`--depth`** (default: `4`)  
  Depth of the model.

- **`--freeze_bn`** (default: `False`)  
  Freeze batch normalization layers during training. 

Anytime a new training experiment is run, a new directory will be created in `--logdir` with the name `{exp_name}`. This directory will contain the model checkpoints and logs for that experiment.

