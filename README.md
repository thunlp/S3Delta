# Sparse Structure Search for Delta Tuning

## Dependencies

```shell
pip install -r requirements.txt
```
Maybe you need to change the version of some libraries depending on your servers.

But notice that: `datasets==2`, otherwise there will be some bugs.

`pytorch` version: refer to https://pytorch.org/get-started/locally/ and choose the version suitable for your server.


## Usage

```shell
CUDA_VISIBLE_DEVICES=0 \
python ./NAS_global/search.py \
    --model_name_or_path t5-large \
    --task_name $TASK_NAME \
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
    --modify_configs_from_json $MODIFY_CONFIGS_FROM_JSON \
    --output_dir $OUTPUT_DIR
```

For example, you can test with the following parameters:

```shell
TASK_NAME="superglue-cb"
SEARCH_TRAIN_BATCH_SIZE=4
SEARCH_VALID_BATCH_SIZE=32
SEARCH_TRAIN_EPOCHS=1
SEARCH_VALID_STEPS=30
DELTA_TUNING_TRAIN_BATCH_SIZE=4
DELTA_TUNING_VALID_BATCH_SIZE=32
DELTA_TUNING_TRAIN_EPOCHS=1
DELTA_TUNING_VALID_STEPS=60
LEARNING_RATE=0.0003
SEED=100
ALPHA_LEARNING_RATE=0.1
MAX_NUM_PARAM_RELATIVE=0.0001389
MODIFY_CONFIGS_FROM_JSON=./NAS_global/structure_config/all/mix_low_rank.json
OUTPUT_DIR=./output/example/${TASK_NAME}
```

The program will be executed in about 7 minutes, using 17.2G of video memory on A100.

The specific parameters for the different datasets are detailed in the paper.

## Reproduction
Scripts listed in the folder `scripts/` can be used to reproduct results of $S^3PET\text{-} L$ & $S^3PET\text{-}M$ with different budgets, e.g. run `sh scripts/S3PET-L/1.39/cola.sh`.

## Structure config

The configuration files for the different structures(LoRa,Adapter,S3Delta-Mix,etc.) are in the **`NAS_global/structure_config`** folder. You can specify one in the parameter `modify_configs_from_json`.

## Tasks

The code to process the data is here:
`NAS_global/examples_seq2seq/data_processors/tasks.py`. GLUE and SuperGLUE have been included. You can add new code for other tasks.

## Visualization

Specify parameters here.

```python
backbone_params = 703.4677734375
sparse_rate_relative = 0.0000348
root_path = "output/example"
seed_list = [100]
datasets = ['superglue-cb']
save_name = f'example_superglue-cb_heatmap.pdf'
```

## Reference

Official implementation of [DARTS](https://github.com/quark0/darts)
