#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/mnt/fengyao.hjj/transformers/src

BASE_DIR=/mnt/fengyao.hjj
PRETRAINED_MODEL=${BASE_DIR}/deberta-v3-base
TRANSFORMERS=${BASE_DIR}/transformers

RR=${BASE_DIR}/rhetorical-role-baseline
DATA=${RR}/datasets/lm/tasks-specific
train_file=${DATA}/train.txt
validation_file=${DATA}/dev.txt
outputs_dir=${RR}/results/deberta_continue_only_tasks_data
num_gpus=4
batch_size=8

TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 -m torch.distributed.launch --nproc_per_node=${num_gpus} \
python3 ${RR}/experiments/run_language_modeling.py \
  --model_name_or_path ${PRETRAINED_MODEL} \
  --output_dir ${outputs_dir} \
  --model_type deberta \
  --do_train \
  --train_file ${train_file} \
  --do_eval \
  --validation_file ${validation_file} \
  --line_by_line \
  --metric_path ${TRANSFORMERS}/src/transformers/utils/accuracy.py \
  --max_seq_length 256 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
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


#   --model_name_or_path ${PRETRAINED_MODEL} \
#   --output_dir ${outputs_dir} \
#   --do_train \
#   --do_eval \
#   --train_file ${train_file} \
#   --validation_file ${validation_file} \
#   --evaluation_strategy steps \
#   --max_seq_length 256 \
#   --warmup_steps 500 \
#   --per_device_train_batch_size ${batch_size} \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir $output_dir \
#   --overwrite_output_dir \
#   --logging_steps 1000 \
#   --logging_dir $output_dir
