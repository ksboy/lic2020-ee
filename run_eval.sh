MAX_LENGTH=256
TASK=trigger
MODEL=/home/mhxia/whou/workspace/pretrained_models/chinese_roberta_wwm_large_ext_pytorch  #albert-xxlarge-v2/  #bert-large-uncased-wwm/
DATA_DIR=./data/trigger_classify_weighted/
SCHEMA=./data/event_schema/event_schema.json
OUTPUT_DIR=./output/trigger_classify/0/
EVAL_BATCH_SIZE=64
SEED=1

CUDA_VISIBLE_DEVICES=2 python3 run_classify.py \
--task $TASK \
--model_type bert \
--model_name_or_path $MODEL \
--do_eval \
--data_dir $DATA_DIR \
--do_lower_case \
--keep_accents \
--schema $SCHEMA \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--seed $SEED \
# --fp16 \
# --freeze 
# --overwrite_cache  \
# --eval_all_checkpoints \
