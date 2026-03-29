#!/bin/bash

export WANDB_API_KEY=

for i in {5..7}
do
    TRAIN_PATH="data/pretrain_${i}.csv"
    
    python train.py \
        --no-test \
        --train_path="$TRAIN_PATH" \
        --exp="pretrain_${i}"
done