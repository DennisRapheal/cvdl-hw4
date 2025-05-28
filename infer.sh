#!/bin/bash
python ./inference.py \
  --ckpt ./train_ckpt/model-10/periodic-epoch=0799.ckpt \
  --input_dir ./hw4_realse_dataset/test/degraded \
  --output_dir ./outputs/fullep800

python ./inference.py \
  --ckpt ./train_ckpt/model-10/periodic-epoch=0399.ckpt \
  --input_dir ./hw4_realse_dataset/test/degraded \
  --output_dir ./outputs/fullep400

python ./inference.py \
  --ckpt ./train_ckpt/model-10/periodic-epoch=0199.ckpt \
  --input_dir ./hw4_realse_dataset/test/degraded \
  --output_dir ./outputs/fullep200

python ./inference.py \
  --ckpt ./train_ckpt/model-10/periodic-epoch=0099.ckpt \
  --input_dir ./hw4_realse_dataset/test/degraded \
  --output_dir ./outputs/fullep100

# python ./inference.py \
#   --ckpt ./train_ckpt/model-9/epoch=399-step=80000.ckpt \
#   --input_dir ./hw4_realse_dataset/test/degraded \
#   --output_dir ./outputs/genblock

# python ./inference.py \
#   --ckpt ./train_ckpt/model-8/epoch=299-step=60000.ckpt \
#   --input_dir ./hw4_realse_dataset/test/degraded \
#   --output_dir ./outputs/promptlen10