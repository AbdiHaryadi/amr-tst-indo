import sys
sys.path.append("./AMRBART-id/fine-tune")

from common.options import DataTrainingArguments, ModelArguments, Seq2SeqTrainingArguments
from data_interface.dataset import AMRParsingDataSet, DataCollatorForAMRParsing
import datasets
from datasets import load_from_disk
from huggingface_hub import snapshot_download
import json
import logging
from model_interface.modeling_bart import MBartForConditionalGeneration as BartForConditionalGeneration
from model_interface.tokenization_bart import AMRBartTokenizer
import os
from seq2seq_trainer import Seq2SeqTrainer
import transformers
from transformers import (
    AutoConfig,
    MBartTokenizer,
    MBartTokenizerFast,
    set_seed
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def mkdir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def prepare_tokenizer_and_model(training_args, model_args, data_args):
    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AMRBartTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    # config dec_start_token, max_pos_embeddings
    if model.config.decoder_start_token_id is None and isinstance(
        tokenizer, (MBartTokenizer, MBartTokenizerFast)
    ):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    if training_args.label_smoothing_factor > 0 and not hasattr(
        model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )
        
    return tokenizer, model

def prepare_data_args(data_path):
    data_cache_parent = f"{data_path}/.cache"
    mkdir_if_not_exists(data_cache_parent)
    data_cache = f"{data_cache_parent}/dump-amrparsing"
    # TODO for future: Export HF_DATASETS_CACHE with data_cache here.
    mkdir_if_not_exists(data_cache)

    data_args = DataTrainingArguments(
        data_dir=data_path,
        unified_input=True,
        train_file=f"{data_path}/inference.jsonl", # Why you need this???
        validation_file=f"{data_path}/inference.jsonl", # Why you need this???
        test_file=f"{data_path}/inference.jsonl",
        data_cache_dir=data_cache,
        overwrite_cache=True,
        max_source_length=400,
        max_target_length=1024
    )
    
    return data_args

def setup_logging(log_level):
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

DEFAULT_ROOT_DIR = os.path.join(os.path.dirname(__file__), "AMRBART-id")
class TextToAMR:
    def __init__(
            self,
            model_name: str,
            root_dir: str = DEFAULT_ROOT_DIR,
            dataset: str = "wrete",
            logging_at_training_process_level: bool = False,
    ):
        # Default root_dir: parent of fine-tune (project root)

        output_dir_parent = f"{root_dir}/outputs"
        mkdir_if_not_exists(output_dir_parent)

        output_dir = f"{output_dir_parent}/infer-{model_name}"
        mkdir_if_not_exists(output_dir)

        batch_size=5
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            do_predict=True,
            logging_dir=f"{output_dir}/logs",
            seed=42,
            dataloader_num_workers=1,
            report_to="tensorboard",
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=True,
            include_inputs_for_metrics=True,

            smart_init=False,
            predict_with_generate=True,
            task="text2amr",
            generation_max_length=1024,
            generation_num_beams=5,

            # Issue: Do we really need this?
            per_device_eval_batch_size=batch_size,
            eval_dataloader_num_workers=1,
        )

        if logging_at_training_process_level:
            setup_logging(self.training_args.get_process_log_level())

        self.model_args = ModelArguments(
            model_name_or_path=f"{root_dir}/models/{model_name}",
            cache_dir=f"{root_dir}/.cache",
            use_fast_tokenizer=False
        )

        data_path_parent = f"{root_dir}/ds"
        mkdir_if_not_exists(data_path_parent)
        data_path = f"{data_path_parent}/{dataset}"
        mkdir_if_not_exists(data_path)
        self.data_args = prepare_data_args(data_path)

        self.tokenizer, self.model = prepare_tokenizer_and_model(self.training_args, self.model_args, self.data_args)

        self.data_collator = DataCollatorForAMRParsing(
            self.tokenizer,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

        self.model_name = model_name

    def __call__(self, sentences: list[str]):
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

        predict_dataset = self._prepare_predict_dataset(sentences)
        max_length = (
            self.training_args.generation_max_length
            if self.training_args.generation_max_length is not None
            else self.data_args.val_max_target_length
        )

        num_beams = (
            self.data_args.num_beams
            if self.data_args.num_beams is not None
            else self.training_args.generation_num_beams
        )

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )

        graphs = self._make_graphs(predict_results.predictions)
        assert len(graphs) == len(sentences), f"Inconsistent lengths for graphs ({len(graphs)}) vs sentences ({len(sentences)})"

        idx = 0
        for gp, snt in zip(graphs, sentences):
            metadata = {}
            metadata["id"] = str(idx)
            metadata["annotator"] = self.model_name
            metadata["snt"] = snt
            if "save-date" in metadata:
                del metadata["save-date"]
            gp.metadata = metadata
            idx += 1

        return graphs

    @staticmethod
    def from_huggingface(repo_id: str, model_name: str, root_dir: str = DEFAULT_ROOT_DIR, hf_kwargs: dict = {}, **kwargs):
        local_dir = f"{root_dir}/models"
        mkdir_if_not_exists(local_dir)
        snapshot_download(repo_id=repo_id, local_dir=local_dir, **hf_kwargs)

        return TextToAMR(model_name, root_dir, **kwargs)
    
    def _prepare_predict_dataset(self, sentences: list[str]):
        data_args = self.data_args
        with open(data_args.test_file, encoding="utf-8", mode="w") as fp:
            for x in sentences:
                json_str = json.dumps({"sent": x, "amr": "", "lang": "id"})
                print(json_str, file=fp)

        raw_datasets = AMRParsingDataSet(self.tokenizer, data_args, self.model_args)
        column_names = raw_datasets.datasets["train"].column_names

        if "test" not in raw_datasets.datasets:
            raise ValueError("--do_predict requires a test dataset")

        predict_dataset = raw_datasets.datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        if data_args.overwrite_cache or not os.path.exists(data_args.data_cache_dir + "/test"):
            with self.training_args.main_process_first(desc="prediction dataset map pre-processing"):
                predict_dataset = predict_dataset.map(
                    raw_datasets.tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )
                predict_dataset.save_to_disk(data_args.data_cache_dir + "/test")
        else:
            predict_dataset = load_from_disk(data_args.data_cache_dir + "/test", keep_in_memory=True)

        return predict_dataset
    
    def _make_graphs(self, preds):
        graphs = []
        for idx in range(len(preds)):
            ith_pred = preds[idx]
            ith_pred[0] = self.tokenizer.bos_token_id
            ith_pred = [
                self.tokenizer.eos_token_id if itm == self.tokenizer.amr_eos_token_id else itm
                for itm in ith_pred if itm != self.tokenizer.pad_token_id
            ]

            graph, status, (lin, backr) = self.tokenizer.decode_amr(
                ith_pred, restore_name_ops=False
            )
            graph.status = status
            graph.nodes = lin
            graph.backreferences = backr
            graph.tokens = ith_pred

            graphs.append(graph)

        return graphs