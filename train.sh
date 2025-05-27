
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
# CUDA_VISIBLE_DEVICES=5,6,0,1 python train.py \
#   --batch_size 8 \
#   --num_gpus 4 \
#   --epochs 200 \
#   --ckpt_dir train_ckpt/model-3

# export WANDB_KEY_ALT="c9e88b4e39eafd7eb1a8cc3b1f5daeeaeb61188e"

# 單次執行
# WANDB_API_KEY=$WANDB_KEY_ALT \
# WANDB_ENTITY=my_alt_entity \   # 如果想把 run 放到其他 entity，可加這行
# python train.py

# prompt_len = 10 , version 1 log
# python train_.py \
#   --batch_size 4 \
#   --num_gpus 4 \
#   --epochs 300 \
#   --ckpt_dir train_ckpt/model-8

# prompt_gen block , version 2 log
CUDA_VISIBLE_DEVICES=5,6,0,3 python train_.py \
  --batch_size 4 \
  --num_gpus 4 \
  --epochs 800 \
  --ckpt_dir train_ckpt/model-10