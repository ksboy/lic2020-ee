MAX_LENGTH=384
TASK=role
MODEL=/home/mhxia/whou/workspace/pretrained_models/chinese_roberta_wwm_large_ext_pytorch  #albert-xxlarge-v2/  #bert-large-uncased-wwm/
DATA_DIR=./data/role_bin/test3/
SCHEMA=./data/event_schema/event_schema.json
OUTPUT_DIR=./output/role_bin_train_dev/5/
EVAL_BATCH_SIZE=32
SEED=1

CUDA_VISIBLE_DEVICES=1 python3 run_bi_ner.py \
--task $TASK \
--model_type bert \
--model_name_or_path $MODEL \
--do_predict \
--data_dir $DATA_DIR \
--do_lower_case \
--keep_accents \
--schema $SCHEMA \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--seed $SEED \
# --fp16 \
# --freeze \
# --overwrite_cache \
