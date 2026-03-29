#!/bin/bash


python train.py \
    --no-test \
    --train_path="data/dev_train_target.csv"\
    --finetune_from="ft_agg_frozen_adaptor_lora_wd0.001" \
    --exp="fewshot_adaptor_lora_wd0.001" \
    --max_lr=1e-8 \
    --min_lr=1e-8 \
    --accumulation_step=1 \
    --epochs=3 \
    --weight_decay=0.01 \
    --warmup_epochs=0 \
    --scheduler=cosine \
    --aggregation \
    --adaptor_lora
