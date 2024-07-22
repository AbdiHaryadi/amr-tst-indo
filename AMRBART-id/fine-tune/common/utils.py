# coding:utf-8
import re
import json
import nltk
import torch
import random
import itertools
import numpy as np
from typing import List, Tuple
from transformers import PreTrainedTokenizer
from typing import List, Tuple
nltk.data.path.append("/mnt/nfs-storage/nltk_data/")


def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    assert nltk, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
    return "\n".join(nltk.sent_tokenize(x))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def copy_weights(src, tgt):
    src_params = dict(src.named_parameters())
    tgt_params = tgt.named_parameters()
    for name, param in tgt_params:
        assert name in src_params, f"{name} not in {[n for n,v in src_params]}"
        param.data = src_params[name].data


def mask_tokens(
    inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability).to(labels.device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8).to(labels.device)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5).to(labels.device)).bool() & masked_indices & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def save_dummy_batch(batch, tokenizer, output_dir):
    json_out_path = open(output_dir + "/dummy_input.json", "w", encoding="utf-8")
    ith_dict = {}
    for k, v in batch.items():
        if "_ids" in k and v is not None:
            ith_dict[k] = str(v.tolist())
            ith_dict[k.replace("ids", "tokens")] = tokenizer.batch_decode(v.tolist(), clean_up_tokenization_spaces=False)
        elif "labels" in k:
            ith_dict[k] = str(v.tolist())
            label_data_new = [[idx if idx != -100 else tokenizer.pad_token_id for idx in ith_label] for ith_label in v.tolist()]
            ith_dict[k+"_tokens"] = tokenizer.batch_decode(label_data_new, clean_up_tokenization_spaces=False)
        else:
            print(f"Skiping {k}...")
    json.dump(ith_dict, json_out_path, indent=4)


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


def smart_emb_init(tokenizer, model):
    print("Initializing AMR Vocab according to similar tokens ...")
    
    for tok, idx in tokenizer.encoder.items():
        tok = tok.lstrip(tokenizer.INIT)

        if idx < tokenizer.old_enc_size:
            continue

        elif tok.startswith("<pointer:") and tok.endswith(">"):
            tok_split = ["pointer", str(tok.split(":")[1].strip(">"))]

        elif tok.startswith("<"):
            continue

        elif tok.startswith(":"):

            if tok.startswith(":op"):
                tok_split = ["relation", "operator", str(int(tok[3:]))]

            elif tok.startswith(":snt"):
                tok_split = ["relation", "sentence", str(int(tok[4:]))]

            elif tok.startswith(":ARG"):
                tok_split = ["relation", "argument", str(int(tok[4:]))]

            else:
                tok_split = ["relation"] + tok.lstrip(":").split("-")

        else:
            tok_split = tok.split("-")

        tok_split_ = tok_split
        tok_split = []
        for s in tok_split_:
            s_ = s + tokenizer.INIT
            if s_ in tokenizer.encoder:
                # print(f"{s_} in tokenizer vocabulary")
                tok_split.append(s_)
            else:
                tok_split.extend(tokenizer._tok_bpe(s))  #

        vecs = []
        for s in tok_split:
            idx_split = tokenizer.encoder.get(s, -1)
            if idx_split > -1:
                vec_split = model.model.shared.weight.data[idx_split].clone()
                vecs.append(vec_split)

        if vecs:
            vec = torch.stack(vecs, 0).mean(0)
            noise = torch.empty_like(vec)
            noise.uniform_(-0.1, +0.1)
            model.model.shared.weight.data[idx] = vec + noise