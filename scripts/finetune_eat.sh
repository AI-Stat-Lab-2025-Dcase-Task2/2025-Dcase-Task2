#!/bin/bash

export WANDB_API_KEY=

python train.py \
    --no-test \
    --devices=0 \
    --train_path="data/dev_train_pseudo_domainX_eat.csv" \
    --finetune_from=eat_pretrain_aggregation \
    --exp=ft_eat_big_lr \
    --max_lr=1e-7 \
    --min_lr=1e-9 \
    --batch_size=32 \
    --accumulation_step=8 \
    --weight_decay=0.01 \
    --aggregation \
    --adaptor_lora \
    --epochs=40 \
    --freeze_extractor 

