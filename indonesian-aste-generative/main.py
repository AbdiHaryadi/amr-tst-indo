import os
import csv
from pprint import pprint
import time

from transformers import T5ForConditionalGeneration, T5Tokenizer

from args import init_args
from src.postprocess import (
    IPostprocess,
    EditDistancePostProcessor,
    EmbeddingDistancePostProcessor,
)
from src.loader import ILoader, HotelLoader
from src.utility import get_config, set_seed
from src.constant import Path, ModelType, PostprocessType, ProcessType
from src.trainer import ITrainer, T5Trainer
from src.generator import IGenerator, T5Generator
from src.evaluation import Evaluator

# == Dependencies Maps (Factory) ==
trainer_config_maps = {ModelType.T5Model: T5Trainer}

tokenizer_config_names = {ModelType.T5Model: T5Tokenizer}

generator_config_names = {ModelType.T5Model: T5Generator}

postprocess_config_names = {
    PostprocessType.EDITDISTANCE: EditDistancePostProcessor,
    PostprocessType.EMBEDDING: EmbeddingDistancePostProcessor,
}


def is_valid_type(type: str) -> bool:
    return type in [ModelType.T5Model]  # TODO: adjust based on experiment scenario...

def print_csv(path):
    with open(path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                print(f'Columns: {", ".join(row)}')
            else:
                print(f'\t{row[0]} | {row[1]} | {row[2]}.')

def load_result(prefix):
    train_path = os.path.join("bin", prefix, "train", "score.csv")
    val_path = os.path.join("bin", prefix, "val", "score.csv")
    test_path = os.path.join("bin", prefix, "test", "score.csv")

    if os.path.isfile(train_path):
        print("Train:")
        print_csv(train_path)

    if os.path.isfile(val_path):
        print("Validation:")
        print_csv(val_path)

    if os.path.isfile(test_path):
        print("Test:")
        print_csv(test_path)

if __name__ == "__main__":

    # 1. Initialize scripts argument + setup + dependencies
    args = init_args()
    print(args)
    config_path = args.config
    prefix = args.prefix
    do_train = args.do_train
    do_test = args.do_test

    # config_path = "resources/t5-config.yaml"
    configs = get_config(config_path)
    set_seed(configs["main"]["seed"])

    mode = configs.get("main").get("mode")

    model_type = configs.get("type")
    if not is_valid_type(model_type):
        raise ValueError("Mode Not Available!")
    model_name = configs.get("main").get("pretrained")
    use_checkpoint = configs.get("trainer").get("use_checkpoint")
    if use_checkpoint:
        model_name = configs.get("trainer").get("checkpoint_path")
    print(f"Tokenizer type: {model_name}")
    tokenizer = tokenizer_config_names.get(model_type).from_pretrained(model_name)

    # 2. Preparing Dataset ...
    loader: ILoader = HotelLoader(tokenizer, configs)

    train_loader, val_loader = loader.get_train_loader(), loader.get_val_loader()
    train_dataset, val_dataset = loader.get_train_dataset(), loader.get_val_dataset()
    train_sents, val_sents = train_dataset.get_sents(), val_dataset.get_sents()

    test_loader = loader.get_test_loader()
    test_dataset = loader.get_test_dataset()
    test_sents = test_dataset.get_sents()

    # Show dataset statistics...
    if do_train:
        print("\n".join(train_dataset.get_stats("Training")))
    print("\n".join(val_dataset.get_stats("Validation")))
    if do_test:
        print("\n".join(test_dataset.get_stats("Testing")))

    # 3. Training (skip if do-test only)
    trainer_fn = trainer_config_maps.get(model_type)
    if do_train:
        trainer: ITrainer = trainer_fn(
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            prefix=prefix,
            configs=configs,
        )
        trainer.fit()
        trainer.save()

    load_trainer: ITrainer = trainer_fn(
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        prefix=prefix,
        configs=configs,
    )

    saved_path = os.path.join(Path.MODEL, prefix)
    load_trainer.load(saved_path)
    model = load_trainer.get_model()

    postprocessor_type = configs.get("normalization").get("mode")
    postprocessor: IPostprocess = postprocess_config_names.get(postprocessor_type)()

    if isinstance(postprocessor, EmbeddingDistancePostProcessor) and isinstance(model, T5ForConditionalGeneration) :
        postprocessor.set_embedding(tokenizer, model.get_input_embeddings())

    # 4. Evaluation ...
    evaluator = Evaluator(postprocessor, configs)
    if do_train:
        train_time = time.time()
        evaluator.export("train", prefix, tokenizer, model, train_loader, train_sents)    
        print("--- Train Generation: %s seconds ---" % (time.time() - train_time))
    if do_test:
        test_time = time.time()
        evaluator.export("test", prefix, tokenizer, model, test_loader, test_sents)
        print("--- Test Generation: %s seconds ---" % (time.time() - test_time))
    eval_time = time.time()
    evaluator.export("val", prefix, tokenizer, model, val_loader, val_sents)
    print("--- Eval Generation: %s seconds ---" % (time.time() - eval_time))
    load_result(prefix)

    # 5. Inference / Generate ... -> only be used for demo only
    sents = [
        "pelayanan ramah , kamar nyaman dan fasilitas lengkap . hanya airnya showernya kurang panas .",
        "tidak terlalu jauh dari pusat kota .",
        "dengan harga terjangkau kita sudah mendapatkan fasilitas yang nyaman .",
        "kamar luas dan bersih . seprai bersih .",
        "seprai nya kurang bersih .",
        "kamarnya bersih dan rapi . saya kebetulan dapat yang di lantai dua .",
    ]

    generator: IGenerator = generator_config_names.get(model_type)(
        tokenizer, model, postprocessor, configs
    )
    res = generator.generate(sents, implicit=False,fix=True)
    pprint(res)


    