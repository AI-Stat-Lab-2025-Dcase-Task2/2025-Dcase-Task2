#!/bin/bash

export WANDB_API_KEY=dac5fb09505bcfc962be146bf45994e9555b921c

for i in {5..7}
do
    TRAIN_PATH="data/pretrain_${i}.csv"
    
    python train.py \
        --no-test \
        --train_path="$TRAIN_PATH" \
        --exp="pretrain_${i}"
done