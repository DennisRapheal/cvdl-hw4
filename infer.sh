#!/bin/bash
python ./inference.py \
  --ckpt ./train_ckpt/model-5/epoch=299-step=60000.ckpt \
  --input_dir ./hw4_realse_dataset/test/degraded \
  --output_dir ./outputs/model-5