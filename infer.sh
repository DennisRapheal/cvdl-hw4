#!/bin/bash
python ./inference.py \
  --ckpt ./train_ckpt/epoch=9-step=3600.ckpt \
  --input_dir ./hw4_realse_dataset/test/degraded \
  --output_dir ./outputs/test-1