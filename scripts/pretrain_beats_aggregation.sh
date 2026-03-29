#!/bin/bash

export WANDB_API_KEY=

python train.py \
    --devices=3 \
    --no-test \
    --train_path="data/pretrain_2.csv" \
    --exp="pretrain_beats_aggregation_2" \
    --audio_model=beats \
    --aggregation \
    --batch_size=16 \
    --accumulation_steps=16
