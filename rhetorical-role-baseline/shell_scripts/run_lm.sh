#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/mnt/fengyao.hjj/transformers/src

BASE_DIR=/mnt/fengyao.hjj
PRETRAINED_MODEL=${BASE_DIR}/zlucia-legalbert
TRANSFORMERS=${BASE_DIR}/transformers

RR=${BASE_DIR}/rhetorical-role-baseline
DATA=${RR}/datasets/lm/tasks-specific-ildc
outputs_dir=${RR}/results/bert_continue_ts_ildc
train_file=${DATA}/train.txt
validation_file=${DATA}/dev.txt

TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
python3 \
  ${RR}/rhetorical-role-baseline/run_language_modeling.py \
  --model_name_or_path ${PRETRAINED_MODEL} \
  --output_dir ${outputs_dir} \
  --model_type bert \
  --do_train \
  --train_file ${train_file} \
  --do_eval \
  --validation_file ${validation_file} \
  --line_by_line \
  --metric_path ${TRANSFORMERS}/src/transformers/utils/accuracy.py \
  --max_seq_length 512 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy 'steps' \
  --num_train_epochs 20 \
  --save_total_limit 8 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --load_best_model_at_end True \
  --overwrite_output_dir False 
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --warmup_steps 1000
#   --ignore_data_skip \
#   --cache_dir ${DATA}
