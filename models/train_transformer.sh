#!/usr/bin/env bash

PROJECT_ROOT=..

CUDA_VISIBLE_DEVICES=0 fairseq-train \
                    $PROJECT_ROOT/data/python_preprocessed_limited_vocab/ \
                    --log-format json \
                    --log-interval 100 \
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
                    --save-dir transformer_checkpoints \
                    --save-interval-updates 10000 \
                    --keep-interval-updates 1
#     --max-sentences 32
#     --max-tokens 32 \
    #     --fp16 \
    #     --memory-efficient-fp16 \
