#!/usr/bin/env bash

     torchrun --master_port=29500 --nproc_per_node=2 train.py \
        --dataset cityscapes \
        --val_dataset cityscapes \
        --arch network.deepv3.DeepR50V3PlusD \
        --city_mode 'train' \
        --lr_schedule poly \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --gblur \
        --color_aug 0.5 \
        --rrotate 0 \
        --max_iter 80000 \
        --bs_mult 4 \
        \
        --sgd \
        --lr_enc 2.5e-3 \
        --weight_decay_enc 1e-3 \
        --lr_dec 2.5e-3 \
        --weight_decay_dec 1e-3 \
        --lr_pre 1e-2 \
        --weight_decay_pre 1e-2 \
        --pre_epoch 5 \
        --bs_mult_pre 4 \
        --momentum_enc 0.9999 \
        --momentum_dec 0.9999 \
        \
        --date 0927 \
        --exp res50_city_sgd_m0.9999_labhfc_pre5 \
        --ckpt ./logs/ \
        --tb_path ./logs/