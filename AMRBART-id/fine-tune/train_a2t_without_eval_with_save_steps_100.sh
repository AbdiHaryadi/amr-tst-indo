export CUDA_VISIBLE_DEVICES=0
RootDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
RootDir=$RootDir/..

DataPath=$RootDir/ds/$2

Model=$1
ModelCache=$RootDir/.cache
DataCache=$DataPath/.cache/dump-amr2text
Epoch=$3

OutputDir=${RootDir}/outputs/$Model-fted

if [ ! -d ${OutputDir} ];then
  mkdir -p ${OutputDir}
fi

export HF_DATASETS_CACHE=$DataCache

if [ ! -d ${DataCache} ];then
  mkdir -p ${DataCache}
fi

lr=1e-6
batch_size=5

HubModelId=${4:-}

python -u main.py \
    --overwrite_output_dir \
    --data_dir $DataPath \
    --task "amr2text" \
    --train_file $DataPath/train.jsonl \
    --output_dir $OutputDir \
    --cache_dir $ModelCache \
    --data_cache_dir $DataCache \
    --model_name_or_path $RootDir/models/$Model \
    --unified_input True \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $batch_size \
    --learning_rate $lr \
    --optim "adamw_hf" \
    --lr_scheduler_type "polynomial" \
    --warmup_steps 200 \
    --num_train_epochs $Epoch \
    --max_source_length 1024 \
    --max_target_length 384 \
    --generation_max_length 380 \
    --generation_num_beams 5 \
    --label_smoothing_factor 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 0 \
    --max_steps -1 \
    --smart_init False \
    --use_fast_tokenizer False \
    --logging_dir $OutputDir/logs \
    --logging_first_step True \
    --logging_steps 20 \
    --save_steps 100 \
    --save_total_limit 1 \
    --seed 42 \
    --dataloader_num_workers 1 \
    --eval_dataloader_num_workers 1 \
    --do_train \
    --ddp_find_unused_parameters False \
    --hub_model_id $HubModelId \
    --hub_strategy "all_checkpoints" \
    --report_to "wandb" \
    --dataloader_pin_memory True 2>&1 | tee $OutputDir/run-$(date +"%Y-%m-%dT%H:%M:%S%:z").log
