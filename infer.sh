#!/bin/bash
python ./inference.py \
  --ckpt ./train_ckpt/model-6/epoch=185-step=37200.ckpt \
  --input_dir ./hw4_realse_dataset/test/degraded \
  --output_dir ./outputs/model-5