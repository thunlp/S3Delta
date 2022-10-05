#!/usr/bin/bash
gpu_list=( 0 )
seed_list=( 0 )

ROOT_PATH=.
TASK_NAME="mnli"
DELTA_STRATEGY="S3PET-L"

SEARCH_TRAIN_BATCH_SIZE=32
SEARCH_VALID_BATCH_SIZE=128
SEARCH_TRAIN_EPOCHS=1
SEARCH_VALID_STEPS=200

DELTA_TUNING_TRAIN_BATCH_SIZE=32
DELTA_TUNING_VALID_BATCH_SIZE=128
DELTA_TUNING_TRAIN_EPOCHS=3
DELTA_TUNING_VALID_STEPS=500

LEARNING_RATE=0.0003
ALPHA_LEARNING_RATE=0.1
MAX_NUM_PARAM_RELATIVE=0.0001389
LR_DECAY=True

MODIFY_CONFIGS_FROM_JSON=${ROOT_PATH}/NAS_global/structure_config/lora_qkvowiwo_rank1.json
OUTPUT_DIR=${ROOT_PATH}/output_search/${DELTA_STRATEGY}_${MAX_NUM_PARAM_RELATIVE}/${TASK_NAME}/

export TRANSFORMERS_OFFLINE=0
export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_DATASETS_CACHE=${ROOT_PATH}/huggingface_cache/datasets

if [ ${#seed_list[*]} != ${#gpu_list[*]} ];
then
    echo length not match
    echo should be ${#sparse_rates[*]}
    echo but find ${#gpu_list[*]}
    exit 0
fi

gpu_idx=0
for SEED in ${seed_list[*]};
do
(
    CUDA_VISIBLE_DEVICES=${gpu_list[$gpu_idx]} python ${ROOT_PATH}/NAS_global/search.py \
        --model_name_or_path t5-large \
        --task_name $TASK_NAME \
        --delta_strategy $DELTA_STRATEGY \
        --search_train_batch_size $SEARCH_TRAIN_BATCH_SIZE \
        --search_valid_batch_size $SEARCH_VALID_BATCH_SIZE \
        --search_train_epochs $SEARCH_TRAIN_EPOCHS \
        --search_valid_steps $SEARCH_VALID_STEPS \
        --delta_tuning_train_batch_size $DELTA_TUNING_TRAIN_BATCH_SIZE \
        --delta_tuning_valid_batch_size $DELTA_TUNING_VALID_BATCH_SIZE \
        --delta_tuning_train_epochs $DELTA_TUNING_TRAIN_EPOCHS \
        --delta_tuning_valid_steps $DELTA_TUNING_VALID_STEPS \
        --learning_rate $LEARNING_RATE \
        --seed $SEED \
        --alpha_learning_rate $ALPHA_LEARNING_RATE \
        --how_to_calc_max_num_param "relative" \
        --max_num_param_relative $MAX_NUM_PARAM_RELATIVE \
        --global_strategy "shift" \
        --shift_detach True \
        --detemining_structure_strategy "p" \
        --lr_decay $LR_DECAY \
        --if_search True \
        --train_after_search True \
        --for_baseline False \
        --modify_configs_from_json $MODIFY_CONFIGS_FROM_JSON \
        --output_dir $OUTPUT_DIR \
        
)&
gpu_idx=$((gpu_idx+1))
done
wait