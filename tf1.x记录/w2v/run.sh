#!/usr/bin/env bash

python train.py \
  --filename='text8.zip' \
  --batch_size=128 \
  --embedding_size=128 \
  --skip_window=2 \
  --num_skips=4 \
  --vocab_size=50000 \
  --valid_size=16 \
  --valid_window=100 \
  --num_sample=64 \
  --train_steps=10001