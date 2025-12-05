
#!/usr/bin/env bash

     torchrun --master_port=29500 --nproc_per_node=2 valid.py \
        --val_dataset gtav \
        --arch network.deepv3.DeepR50V3PlusD \
        --enc_snapshot ckpt/res50_city_sgd_labhfc/enc_best_ckpt_city.pth \
        --dec_snapshot ckpt/res50_city_sgd_labhfc/dec_best_ckpt_city.pth \
        --date 0926 \
        --exp res50_city_valid \
        --ckpt ./logs/ \
        --tb_path ./logs/ \
        --wandb_name res50_city_valid