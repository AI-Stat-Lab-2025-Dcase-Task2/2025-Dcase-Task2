#!/bin/bash

export WANDB_API_KEY=dac5fb09505bcfc962be146bf45994e9555b921c

python train.py \
    --no-test \
    --devices=1 \
    --train_path="data/dev_train_pseudo_domainX_beats.csv" \
    --finetune_from=beats_pretrain_aggregation \
    --exp=ft_beats \
    --max_lr=1e-7 \
    --min_lr=1e-9 \
    --batch_size=16 \
    --accumulation_step=16 \
    --weight_decay=0.01 \
    --aggregation \
    --adaptor_lora \
    --epochs=100 \
    --freeze_extractor  \
    --audio_model=beats

