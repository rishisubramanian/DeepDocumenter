#!/usr/bin/env bash

PROJECT_ROOT=..

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    $PROJECT_ROOT/data/python_preprocessed/ \
    --arch lstm \
    --max-epoch 10 \
    --batch-size 32 \
    --log-format tqdm \
    --save-dir seq2seq_checkpoints
