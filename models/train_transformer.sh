#!/usr/bin/env bash

PROJECT_ROOT=..

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    $PROJECT_ROOT/data/python_preprocessed_limited_vocab/ \
    --tensorboard-logdir transformer_tensorboard \
    --skip-invalid-size-inputs-valid-test \
    --arch transformer \
    --optimizer adam \
    --learning-rate 0.001 \
    --lr-scheduler fixed \
    --max-epoch 10 \
    --encoder-layers 2 \
    --decoder-layers 2 \
    --encoder-embed-dim 128 \
    --decoder-embed-dim 128 \
    --encoder-ffn-embed-dim 256 \
    --decoder-ffn-embed-dim 256 \
    --save-dir transformer_checkpoints 
#     --max-sentences 32
#     --max-tokens 32 \
#     --fp16 \
#     --memory-efficient-fp16 \ 