#!/usr/bin/env bash

args=(
  --exp_name "test_run"
  --batch_size 2
  --num_data -1
  --epochs 2
  --warmup_epochs 5
  --peak_lr 1e-3
  --start_lr 1e-4
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
  --base_channels 64
  --depth 4
  --log_interval 1
)

export FORCE_CPU=1
python scripts/train.py "${args[@]}"