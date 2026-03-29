#!/bin/bash

export WANDB_API_KEY=

python train.py \
    --no-test \
    --train_path="data/pretrain_2.csv" \
    --exp="pretrain_eat_aggregation_2" \
    --aggregation
