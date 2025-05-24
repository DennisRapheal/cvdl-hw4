#!/bin/bash
python ./inference.py \
  --ckpt ./train_ckpt/model-2/epoch=199-step=18000.ckpt \
  --input_dir ./hw4_realse_dataset/test/degraded \
  --output_dir ./outputs/model-2