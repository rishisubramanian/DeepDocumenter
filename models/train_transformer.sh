#!/usr/bin/env bash

PROJECT_ROOT=..

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    $PROJECT_ROOT/data/python_preprocessed/ \
    --arch transformer \
    --max-epoch 10 \
    --encoder-layers 2 \
    --decoder-layers 2 \
    --encoder-embed-dim 128 \
    --decoder-embed-dim 128 \
    --encoder-ffn-embed-dim 256 \
    --decoder-ffn-embed-dim 256 \
    --batch-size 8 \
    --log-format tqdm \
    --save-dir transformer_checkpoints
