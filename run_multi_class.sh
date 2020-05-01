
# "财经/交易 产品行为 交往 竞赛行为 人生 司法行为 灾害/意外 组织关系 组织行为"
TaskList="组织关系 组织行为"
for TASK in $TaskList;
do
echo $Task;

MAX_LENGTH=256
MODEL=/home/mhxia/whou/workspace/pretrained_models/chinese_roberta_wwm_large_ext_pytorch  #albert-xxlarge-v2/  #bert-large-uncased-wwm/
DATA_DIR=./data/$TASK
SCHEMA=./data/event_schema/event_schema.json
OUTPUT_DIR=./output/$TASK
BATCH_SIZE=8
EVAL_BATCH_SIZE=64
NUM_EPOCHS=10
SAVE_STEPS=60
# SAVE_STEPS= $save_steps* gradient_accumulation_steps * batch_size * num_gpus
WARMUP_STEPS=80
SEED=1
LR=3e-5

CUDA_VISIBLE_DEVICES=0 python3 run_ner.py \
--task $TASK \
--model_type bert \
--model_name_or_path $MODEL \
--evaluate_during_training \
--eval_all_checkpoints \
--data_dir $DATA_DIR \
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
--do_train \
--do_eval \
# --fp16 
# --freeze 
# --overwrite_cache  \

done