#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/mnt/fengyao.hjj/transformers/src

BASE_DIR=/mnt/fengyao.hjj
PRETRAINED_MODEL=${BASE_DIR}/scibert_scivocab_uncased
TRANSFORMERS=${BASE_DIR}/transformers

RR=${BASE_DIR}/rhetorical-role-baseline
DATA=${RR}/datasets/legaleval
outputs_dir=${RR}/results/scibert_baseline_wo_context
train_file=${DATA}/train.csv
validation_file=${DATA}/dev.csv
test_file=${DATA}/test.csv

TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
/mnt/fengyao.hjj/miniconda3/envs/ugc/bin/python3 \
  ${RR}/experiments/bert_wo_context.py \
  --model_name_or_path ${PRETRAINED_MODEL} \
  --output_dir ${outputs_dir} \
  --do_train \
  --do_eval \
  --do_predict \
  --train_file ${train_file} \
  --validation_file ${validation_file} \
  --test_file ${test_file} \
  --remove_unused_columns True \
  --evaluation_strategy 'steps' \
  --num_train_epochs 8 \
  --save_total_limit 8 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --load_best_model_at_end True \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 512 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --warmup_steps 1000 \
  --overwrite_output_dir False \
  --ignore_data_skip \
  --metric_path ${TRANSFORMERS}/src/transformers/utils/accuracy.py \
  --cache_dir ${DATA}

