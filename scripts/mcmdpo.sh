#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
echo "pythonpath="$PYTHONPATH
sleep 1

export CUDA_VISIBLE_DEVICES="0,1,2,3"

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path  {sft-ckpt} \
    --version v1 \
    --data_path {PAlt-8k} \
    --image_folder / \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 35 \
    --save_total_limit 10 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --output_dir {output_ckpt} \
    --report_to wandb \
    --task DPO \
    --use_image_type diffusion \
    --diffusion_step 700 \
    --use_cl_reward_num 7 \
    --use_cross_modal_loss True \
    --para_norm 2.8 \
    --para_r3 1.0 \
    --para_r1 0.5 \
    --para_r2 0.5 \
    --para_r4 0.2 \
    --para_r5 0.2 \
    --para_r6 0.2 \
    --para_r7 0.2

