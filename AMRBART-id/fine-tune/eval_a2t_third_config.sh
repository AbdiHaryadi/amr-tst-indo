export CUDA_VISIBLE_DEVICES=0
RootDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
RootDir=$RootDir/..

DataPath=$RootDir/ds/$2

Model=$1
ModelCache=$RootDir/.cache
DataCache=$DataPath/.cache/dump-amr2text

OutputDir=${RootDir}/outputs/eval-$Model

if [ ! -d ${OutputDir} ];then
  mkdir -p ${OutputDir}
fi

export HF_DATASETS_CACHE=$DataCache

if [ ! -d ${DataCache} ];then
  mkdir -p ${DataCache}
fi

batch_size=2

python -u main.py \
    --overwrite_output_dir \
    --use_lang_prefix \
    --data_dir $DataPath \
    --task "amr2text" \
    --validation_file $DataPath/dev.jsonl \
    --output_dir $OutputDir \
    --cache_dir $ModelCache \
    --data_cache_dir $DataCache \
    --overwrite_cache True \
    --model_name_or_path $RootDir/models/$Model \
    --unified_input True \
    --per_device_eval_batch_size $batch_size \
    --max_source_length 1024 \
    --max_target_length 384 \
    --generation_max_length 380 \
    --generation_num_beams 5 \
    --predict_with_generate \
    --smart_init False \
    --use_fast_tokenizer False \
    --logging_dir $OutputDir/logs \
    --seed 42 \
    --dataloader_num_workers 1 \
    --eval_dataloader_num_workers 1 \
    --metric_for_best_model "eval_bleu" \
    --include_inputs_for_metrics \
    --do_eval \
    --ddp_find_unused_parameters False \
    --dataloader_pin_memory True 2>&1 | tee $OutputDir/run.log