export CUDA_VISIBLE_DEVICES=0
RootDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
RootDir=$RootDir/..

ValidationPath=$2

Model=$1
ModelCache=$RootDir/.cache
DataCache=$DataPath/.cache/dump-amrparsing

OutputDir=${RootDir}/outputs/loss-eval-$Model

if [ ! -d ${OutputDir} ];then
  mkdir -p ${OutputDir}
fi

export HF_DATASETS_CACHE=$DataCache

if [ ! -d ${DataCache} ];then
  mkdir -p ${DataCache}
fi

lr=1e-6
batch_size=5

python -u main.py \
    --overwrite_output_dir \
    --data_dir $DataPath \
    --task "text2amr" \
    --validation_file $ValidationPath \
    --output_dir $OutputDir \
    --cache_dir $ModelCache \
    --data_cache_dir $DataCache \
    --model_name_or_path $RootDir/models/$Model \
    --unified_input True \
    --per_device_eval_batch_size $batch_size \
    --max_source_length 400 \
    --max_target_length 1024 \
    --generation_max_length 1024 \
    --generation_num_beams 5 \
    --smart_init False \
    --use_fast_tokenizer False \
    --logging_dir $OutputDir/logs \
    --seed 42 \
    --dataloader_num_workers 1 \
    --eval_dataloader_num_workers 1 \
    --do_eval \
    --ddp_find_unused_parameters False \
    --dataloader_pin_memory True 2>&1 | tee $OutputDir/run-$(date +"%Y-%m-%dT%H:%M:%S%:z").log
