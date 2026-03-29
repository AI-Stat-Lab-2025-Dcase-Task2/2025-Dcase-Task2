#!/bin/bash

export WANDB_API_KEY=dac5fb09505bcfc962be146bf45994e9555b921c

python train.py \
    --no-test \
    --train_path="data/pretrain_2.csv" \
    --exp="pretrain_eat_aggregation_2" \
    --aggregation
