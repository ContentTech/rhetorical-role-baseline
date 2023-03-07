#!/bin/bash
# 训练transformers-based模型
# 安装transformers或者 export PYTHONPATH= $PYTHONPATH:/mnt/fengyao.hjj/transformers/src
export PYTHONPATH= $PYTHONPATH:/mnt/fengyao.hjj/transformers/src

#  --model_name_or_path /mnt/fengyao.hjj/rhetorical-role-baseline/results/bert_continue_bz64/checkpoint-48000
#   --model_name_or_path /mnt/fengyao.hjj/zlucia-legalbert \

TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
python \
  /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/run_bert_wo_context.py \
  --model_name_or_path /mnt/fengyao.hjj/rhetorical-role-baseline/results/bert_continue_bz64/checkpoint-48000 \
  --output_dir /mnt/fengyao.hjj/rhetorical-role-baseline/results/bert_continue_bz64_baseline_wo_context_D0 \
  --do_train \
  --do_eval \
  --do_predict \
  --train_file /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D0/train.csv \
  --validation_file /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D0/dev.csv \
  --test_file /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D0/test.csv \
  --remove_unused_columns True \
  --evaluation_strategy 'steps' \
  --num_train_epochs 3 \
  --save_total_limit 3 \
  --eval_steps 100 \
  --save_steps 100 \
  --load_best_model_at_end True \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 512 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --warmup_steps 1000 \
  --overwrite_output_dir False \
  --ignore_data_skip \
  --metric_path /mnt/fengyao.hjj/text_matching/src/transformers/utils/accuracy.py \
  --cache_dir /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/ 2>&1 | tee /mnt/fengyao.hjj/rhetorical-role-baseline/results/bert_continue_bz64_baseline_wo_context_D0.log
