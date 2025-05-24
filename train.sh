
clear
# model-1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
#   --batch_size 16 \
#   --num_gpus 4 \
#   --epochs 80 \
#   --ckpt_dir train_ckpt/model-1

# model-2 original, prompt_len = 5
# CUDA_VISIBLE_DEVICES=0,1,6,7 python train.py \
#   --batch_size 8 \
#   --num_gpus 4 \
#   --epochs 200 \
#   --ckpt_dir train_ckpt/model-2


# model-2 original, prompt_len = 2
CUDA_VISIBLE_DEVICES=5,6,0,1 python train.py \
  --batch_size 8 \
  --num_gpus 4 \
  --epochs 200 \
  --ckpt_dir train_ckpt/model-3