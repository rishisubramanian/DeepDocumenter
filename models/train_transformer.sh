#!/usr/bin/env bash

PROJECT_ROOT=..

fairseq-train \
    $PROJECT_ROOT/data/python_preprocessed/ \
    --arch transformer \
    --max-epoch 10 \
    --encoder-layers 2 \
    --decoder-layers 2 \
    --encoder-embed-dim 256 \
    --decoder-embed-dim 256 \
    --encoder-ffn-embed-dim 512 \
    --decoder-ffn-embed-dim 512 \
    --batch-size 16 \
    --log-format tqdm \
    --save-dir transformer_checkpoints