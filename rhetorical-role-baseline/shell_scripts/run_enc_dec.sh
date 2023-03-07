#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/mnt/fengyao.hjj/transformers/src

CUDA_VISIBLE_DEVICES=0 \
python \
/mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/run_seq2seq.py \
  --model_name_or_path /mnt/fengyao.hjj/pretrained_models/t5-base/ \
  --output_dir  /mnt/fengyao.hjj/rhetorical-role-baseline/results/2_result_enc_dec_1221/t5_d4_window9 \
  --source_prefix 'summarize: ' \
  --do_train \
  --do_eval \
  --do_predict \
  --train_file /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/enc_dec/D4/train_enc_dec_window9.csv \
  --validation_file /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/enc_dec/D4/dev_enc_dec_window9.csv \
  --test_file /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/enc_dec/D4/dev_enc_dec_window9.csv \
  --text_column 'context' \
  --summary_column 'label' \
  --evaluation_strategy 'steps' \
  --num_train_epochs 10 \
  --save_total_limit 5 \
  --eval_steps 100 \
  --save_steps 100 \
  --load_best_model_at_end True \
  --max_source_length 4096 \
  --resize_position_embeddings True \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_beams 12 \
  --remove_unused_columns False \
  --predict_with_generate True \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --warmup_steps 1000 \
  --metric_path /mnt/fengyao.hjj/transformers/src/transformers/utils/sacrebleu.py



# CUDA_VISIBLE_DEVICES=0 \
# python \
# /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/run_seq2seq.py \
#   --model_name_or_path /mnt/fengyao.hjj/rhetorical-role-baseline/results/2_result_enc_dec_1221/t5_d4_chunk9/checkpoint-29100 \
#   --output_dir /mnt/fengyao.hjj/rhetorical-role-baseline/results/2_result_enc_dec_1221/t5_d4_chunk9/checkpoint-29100 \
#   --source_prefix 'summarize: ' \
#   --text_column 'context' \
#   --summary_column 'label' \
#   --do_predict \
#   --train_file /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/enc_dec/D4/train_enc_dec_chunk9.csv \
#   --validation_file /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/enc_dec/D4/dev_enc_dec_chunk9.csv \
#   --test_file /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/enc_dec/D4/dev_enc_dec_chunk9.csv \
#   --num_beams 12 \
#   --per_device_eval_batch_size 4 \
#   --remove_unused_columns False \
#   --predict_with_generate True \
#   --metric_path /mnt/fengyao.hjj/transformers/src/transformers/utils/sacrebleu.py

