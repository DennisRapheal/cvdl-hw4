
clear
# model-1
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --batch_size 16 \
  --num_gpus 4 \
  --epochs 80 \
  --ckpt_dir train_ckpt/model-1