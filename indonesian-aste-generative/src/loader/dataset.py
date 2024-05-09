from torch.utils.data import Dataset
from copy import deepcopy
import numpy as np
from src.loader.interface import IDataset
from src.constant import GENERAL_ASPECT

from src.utility import extract

senttag2word = {"POS": "positif", "NEG": "negatif", "NEU": "netral"}


class HotelDataset(Dataset, IDataset):
    """
    Construct PyTorch dataset object for Indonesian Hotel Review Text Dataset
    sents, labels, paradigm, tokenizer, max_length=128
    Args :
        sents -> List[str] : collenction of sentences as inputs
        labels -> List[str] : collection of targets depending on paradigm
        tokenizer -> Tokenizer: Transformer Tokenizer for T5
        max_length -> int: maximum word length per instance

    Typical usage example:
    dataset = HotelDataset(sents, labels, paradigm, tokenizer)

    or

    params = {
        "sents": sents,
        "labels": labels,
        "tokenizer" : tokenizer,
        "max_length" : 128,
    }
    dataset = HotelDataset(**params)
    """

    def __init__(self, sents, labels, tokenizer, configs):
        self.tokenizer = tokenizer
        self.sents = sents
        self.labels = labels
        self.configs = configs

        self.max_length = configs.get("loader").get("max_seq_length")

        self.inputs = []
        self.targets = []

        self.__build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }

    def get_sents(self):
        return self.sents

    def get_stats(self, name, idx=[]):

        usefilter = len(idx) > 0

        stats_accumulator = []
        stats_accumulator.append("\nDataset: " + name)
        stats_accumulator.append(
            "--------------------------------------------------------"
        )
        stats_accumulator.append(
            "Details                          Value                     "
        )
        stats_accumulator.append(
            "========================================================"
        )

        stats_accumulator.append("%-32s %-12s" % ("Total Instance", len(self.sents)))
        targets = deepcopy(self.extracted_labels)
        if usefilter:
            targets = [datum for i, datum in enumerate(targets) if i in idx]

        triplets_count = np.array([len(extract(triplets)) for triplets in targets])
        stats_accumulator.append(
            "%-32s %-12s" % ("Total Triplet Instance", triplets_count.sum())
        )
        stats_accumulator.append(
            "%-32s %-12s" % ("Triplet avg (per review)", triplets_count.mean())
        )
        stats_accumulator.append(
            "%-32s %-12s" % ("Triplet min (per review)", triplets_count.min())
        )
        stats_accumulator.append(
            "%-32s %-12s" % ("Triplet max (per review)", triplets_count.max())
        )

        sents = deepcopy(self.sents)
        if usefilter:
            sents = [datum for i, datum in enumerate(sents) if i in idx]

        sequence_count = np.array([len(sent) for sent in sents])
        stats_accumulator.append(
            "%-32s %-12s" % ("Sequence length max", sequence_count.max())
        )
        stats_accumulator.append(
            "%-32s %-12s" % ("Sequence length min", sequence_count.min())
        )
        stats_accumulator.append(
            "%-32s %-12s" % ("Sequence length avg", sequence_count.mean())
        )

        def count_unique_words(sents):
            sents = set(sents)
            return len(sents)

        sequence_unique_count = np.array([count_unique_words(sent) for sent in sents])
        stats_accumulator.append(
            "%-32s %-12s" % ("Unique sequence length max", sequence_unique_count.max())
        )
        stats_accumulator.append(
            "%-32s %-12s" % ("Unique sequence length min", sequence_unique_count.min())
        )
        stats_accumulator.append(
            "%-32s %-12s" % ("Unique sequence length avg", sequence_unique_count.mean())
        )

        stats_accumulator.append(
            "--------------------------------------------------------"
        )
        return stats_accumulator

    def __build(self):
        func = generate_extraction_style_target
        inputs = deepcopy(self.sents)
        targets = func(self.sents, self.labels)
        self.extracted_labels = targets
        for i in range(len(inputs)):
            input_item = " ".join(inputs[i])
            tokenized_input = self.tokenizer.batch_encode_plus(
                [input_item],  # TODO: could be optimized...
                max_length=self.configs.get("loader").get("max_seq_length"),
                padding=self.configs.get("loader").get("padding"),
                truncation=self.configs.get("loader").get("truncation"),
                return_tensors="pt",
            )

            target_item = targets[i]
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target_item],
                max_length=self.configs.get("loader").get("max_seq_length"),
                padding=self.configs.get("loader").get("padding"),
                truncation=self.configs.get("loader").get("truncation"),
                return_tensors="pt",
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


def generate_extraction_style_target(sents_e, labels):
    """Helper func for generating target for GAS extraction-style paradigm"""
    extracted_targets = []
 
    for i, label in enumerate(labels):
        all_tri = []
        for tri in label:
            if len(tri[0]) == 1:
                if tri[0][0] == -1: # implicit
                    aspect = GENERAL_ASPECT 
                else:
                    aspect = sents_e[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                aspect = " ".join(sents_e[i][start_idx : end_idx + 1])
            if len(tri[1]) == 1:
                sentiment = sents_e[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                sentiment = " ".join(sents_e[i][start_idx : end_idx + 1])
            polarity = senttag2word[tri[2]]
            all_tri.append((aspect, sentiment, polarity))
        label_strs = ["(" + ", ".join(l) + ")" for l in all_tri]
        extracted_targets.append("; ".join(label_strs))
    
    assert len(sents_e) == len(labels) == len(extracted_targets)
    return extracted_targets
