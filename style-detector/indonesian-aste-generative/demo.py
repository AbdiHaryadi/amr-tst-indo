from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)
import torch
import yaml

from src.utility import set_seed, get_device

from src.postprocess import (
    IPostprocess,
    EditDistancePostProcessor,
    EmbeddingDistancePostProcessor,
)
from src.constant import ModelType, PostprocessType
from src.generator import IGenerator, T5Generator

# == Dependencies Maps (Factory) ==
generator_config_names = {ModelType.T5Model: T5Generator}
postprocess_config_names = {
    PostprocessType.EDITDISTANCE: EditDistancePostProcessor,
    PostprocessType.EMBEDDING: EmbeddingDistancePostProcessor,
}

def predict(generator, sent, implicit=False, fix=True):
    res = generator.generate([sent], implicit=implicit,fix=fix)
    data = res[0]
    return data

def build_generator(configs, path):
    set_seed(configs["main"]["seed"])
    device = get_device()
    model_type = configs.get("type")
    model_name = configs.get("main").get("pretrained")
    use_checkpoint = configs.get("trainer").get("use_checkpoint")
    if use_checkpoint:
        model_name = configs.get("trainer").get("checkpoint_path")
    print(model_name)
    print(path)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    checkpoint = torch.load(path, map_location=device)
    model = T5ForConditionalGeneration.from_pretrained(model_name, ignore_mismatched_sizes=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)

    postprocessor_type = configs.get("normalization").get("mode")
    postprocessor: IPostprocess = postprocess_config_names.get(postprocessor_type)()

    if isinstance(postprocessor, EmbeddingDistancePostProcessor) and isinstance(
        model, T5ForConditionalGeneration
    ):
        postprocessor.set_embedding(tokenizer, model.get_input_embeddings())

    generator: IGenerator = generator_config_names.get(model_type)(
        tokenizer, model, postprocessor, configs
    )
    return generator

def load_generator(config_path, model_path):
    configs = yaml.safe_load(open(config_path, "r"))
    generator = build_generator(configs, model_path)

    def f(sent):
        return predict(generator, sent)
    
    return f
