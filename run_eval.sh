MAX_LENGTH=256
TASK=role
MODEL=/home/mhxia/whou/workspace/pretrained_models/chinese_roberta_wwm_large_ext_pytorch  #albert-xxlarge-v2/  #bert-large-uncased-wwm/
DATA_DIR=./data/role/
SCHEMA=./data/event_schema/event_schema.json
OUTPUT_DIR=./output/role_smooth/
EVAL_BATCH_SIZE=64
NUM_EPOCHS=7
SAVE_STEPS=300
# SAVE_STEPS= $save_steps* gradient_accumulation_steps * batch_size * num_gpus
WARMUP_STEPS=1000
SEED=1
LR=3e-5

CUDA_VISIBLE_DEVICES=3 python3 run_ner.py \
--task $TASK \
--model_type bert \
--model_name_or_path $MODEL \
--do_eval \
--data_dir $DATA_DIR \
--schema $SCHEMA \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--seed $SEED \
# --fp16 \
# --freeze 
# --overwrite_cache  \
