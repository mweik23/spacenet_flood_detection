#!/bin/zsh

args=(
  --exp_name "test_run"
  --batch_size 2
  --num_data -1
  --epochs 2
  --warmup_epochs 5
  --log_interval 2
  --patience 3
  --reduce_factor 0.5
)

export FORCE_CPU=1
python scripts/train.py $args