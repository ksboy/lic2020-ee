MAX_LENGTH=384
TASK=role
MODEL=/home/mhxia/whou/workspace/pretrained_models/chinese_roberta_wwm_large_ext_pytorch  #albert-xxlarge-v2/  #bert-large-uncased-wwm/
DATA_DIR=./data/trigger_role/
SCHEMA=./data/event_schema/event_schema.json
OUTPUT_DIR=./output/role_trigger_384/
BATCH_SIZE=4
EVAL_BATCH_SIZE=64
NUM_EPOCHS=20
SAVE_STEPS=300
# SAVE_STEPS= $save_steps* gradient_accumulation_steps * batch_size * num_gpus
WARMUP_STEPS=1000
SEED=1
LR=3e-5

CUDA_VISIBLE_DEVICES=1,0,2,3 python3 run_ner.py \
--task $TASK \
--model_type bert \
--model_name_or_path $MODEL \
--do_train \
--do_eval \
--evaluate_during_training \
--data_dir $DATA_DIR \
--overwrite_cache  \
--do_lower_case \
--keep_accents \
--schema $SCHEMA \
--output_dir $OUTPUT_DIR \
--overwrite_output_dir \
--max_seq_length  $MAX_LENGTH \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--gradient_accumulation_steps 1 \
--save_steps $SAVE_STEPS \
--logging_steps $SAVE_STEPS \
--num_train_epochs $NUM_EPOCHS \
--early_stop 3 \
--learning_rate $LR \
--weight_decay 0 \
--warmup_steps $WARMUP_STEPS \
--seed $SEED \
# --fp16 \
# --freeze 
# --overwrite_cache  \
# --eval_all_checkpoints \