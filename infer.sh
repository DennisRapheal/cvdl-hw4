#!/bin/bash
python ./inference.py \
  --ckpt ./train_ckpt/model-6/epoch=499-step=100000.ckpt \
  --input_dir ./hw4_realse_dataset/test/degraded \
  --output_dir ./outputs/ep500

# python ./inference.py \
#   --ckpt ./train_ckpt/model-9/epoch=399-step=80000.ckpt \
#   --input_dir ./hw4_realse_dataset/test/degraded \
#   --output_dir ./outputs/genblock

# python ./inference.py \
#   --ckpt ./train_ckpt/model-8/epoch=299-step=60000.ckpt \
#   --input_dir ./hw4_realse_dataset/test/degraded \
#   --output_dir ./outputs/promptlen10
