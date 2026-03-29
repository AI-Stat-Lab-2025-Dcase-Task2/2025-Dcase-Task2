#!/bin/bash

export WANDB_API_KEY=

python train.py \
    --devices=0 \
    --no-test \
    --train_path="data/pretrain_2_target.csv" \
    --exp="pretrain_beats" \
    --audio_model=beats \
    --batch_size=16 \
    --accumulation_steps=16
