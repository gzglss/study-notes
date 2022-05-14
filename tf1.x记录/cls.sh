#!/usr/bin/env bash

export train_path='data/train.csv'
export dev_path='data/dev.csv'

python tf1-分类例子.py \
  --train_path=${train_path} \
  --dev_path=${dev_path} \
  --batch_size=64 \
  --epoch=100 \
  --do_train=True \
  --do_eval=True