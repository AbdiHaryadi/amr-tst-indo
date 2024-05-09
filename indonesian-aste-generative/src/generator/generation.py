from typing import List

from src.generator.interface import IGenerator
from src.utility import get_device, extract


class Generator(IGenerator):
    def __init__(self, tokenizer, model, postprocessor, configs):
        self.device = get_device()
        self.model = model
        self.tokenizer = tokenizer
        self.postprocessor = postprocessor
        self.configs = configs


class T5Generator(Generator):
    def __init__(self, tokenizer, model, postprocessor, configs):
        super().__init__(tokenizer, model, postprocessor, configs)

    def generate(self, sents: List[str], implicit:bool=False, fix:bool=False) -> List[str]:

        # --- [Preprocessing] ---
        splitted_sents = [txt.split(" ") for txt in sents]
        max_length = self.configs.get("loader").get("max_seq_length")

        # --- [Tokenization] ---
        batch = self.tokenizer.batch_encode_plus(
            sents,
            max_length=max_length,
            padding=self.configs.get("loader").get("padding"),
            truncation=self.configs.get("loader").get("truncation"),
            return_tensors="pt",
        )

        # --- [Generating Triplets Opinion] ---
        self.model.eval()
        outs = self.model.generate(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            max_length=max_length,
            num_beams=self.configs.get("generator").get("num_beams"),
            early_stopping=True,
        )

        # --- [Decoding] ---
        outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

        # --- [Normalization, if needed] ---
        all_preds = []
        for out in outputs:
            all_preds.append(extract(out))
        if fix:
            all_preds = self.postprocessor.check_and_fix_preds(all_preds, splitted_sents, implicit=implicit)
        outputs = all_preds
        return outputs
