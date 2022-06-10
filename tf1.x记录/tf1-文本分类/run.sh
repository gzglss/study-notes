#!/usr/bin/env bash

export ptm_path='/fds/pretrain_model/'
export save_path='/fds/model/text-cls'
export data_path='/fds/data/tnews-cls/'

batch_size=256
epochs=3
max_seq_len=64
save_checkpoints_steps=500
num_filter=64
init_lr=0.0001

cloudml jobs submit \
  -c 8 -M 16G -g 1 \
  -n text-cls-${batch_size}-${epochs}-0610-4 \
  -m code.run_train \
  -u fds://gzg/code_package/text-cls-1.0.tar.gz \
  -fe cnbj1-fds.api.xiaomi.net \
  -fb gzg \
  -d cr.d.xiaomi.net/qijianwei/tensorflow-gpu:1.15.0-xm1.0.0-py3-lexicon \
  -gt v100 \
  -gm 32g \
  --priority_class best-effort \
  -a "\
      --ptm_path=$ptm_path \
      --data_path=$data_path \
      --save_path=$save_path \
      --train_batch_size=$batch_size \
      --eval_batch_size=$batch_size \
      --init_learning_rate=$init_lr \
      --epochs=$epochs \
      --max_seq_len=$max_seq_len \
      --save_checkpoints_steps=$save_checkpoints_steps \
      --filter_sizes=[3,4,5] \
      --num_filter=$num_filter
      "
rm *.yaml