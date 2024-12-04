import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "indonesian-aste-generative"))

from demo import load_generator
import stanza
import string
import torch
from transformers import RobertaPreTrainedModel, RobertaTokenizer
from typing import Callable

class StyleDetectorBase:
    """
    Base class for any style detector class
    """

    def __call__(self, text: str, verbose: bool = False) -> list[str]:
        """
        Detect style words from `text`.

        Args:
        - `text`: Text input.

        - `verbose`: If it's True, print additional informations in the process.
        """
        raise NotImplementedError

class StyleDetector(StyleDetectorBase):
    """
    Class for style detector implementation. The detection uses [William's opinion triplet extraction](https://github.com/William9923/indonesian-aste-generative/tree/master).
    """

    def __init__(self, config_path: str, model_path: str, *, lang: str = "id"):
        """
        Initialize `StyleDetector` class.

        Args:
        - `config_path`: Model configuration path, chosen from `indonesian/aste-generative/resources`.
        This config is consistent with train configuration, related to the model referred by
        `model_path`.

        - `model_path`: Model path. The model can be trained with [William's repository](https://github.com/William9923/indonesian-aste-generative/tree/master).

        - `lang`: Supported language. Indonesia (`id`) is the default.
        """
        self.generator = load_generator(config_path, model_path)
        stanza.download(lang=lang)
        self.nlp = stanza.Pipeline(lang=lang, processors="tokenize,mwt,pos,lemma,depparse", tokenize_pretokenized=True)

    def __call__(
            self,
            text: str,
            verbose: bool = False,
            triplets_callback: Callable[[list], None] | None = None 
    ) -> list[str]:
        """
        Detect style words from `text`.

        Args:
        - `text`: Text input.

        - `verbose`: If it's True, print additional informations in the process.

        - `triplets_callback`: If it's not None, this callback will be called and includes involved triplets during the pass.
        """
        data = self.get_triplets(text, verbose)
        if triplets_callback is not None:
            triplets_callback(data)

        if verbose:
            for x in data:
                print("Triplet:", x)
        
        results: list[str] = self.get_style_words_from_triplets(data, verbose)
        return results

    def get_triplets(
            self,
            text: str,
            verbose: bool = False
    ):
        preprocessed_text = self._preprocess_text(text)
        if verbose:
            print("Preprocessed text result:", preprocessed_text)

        data = self.generator(preprocessed_text)
        return data
    
    def get_style_words_from_triplets(
            self,
            triplets: list,
            verbose: bool = False
    ):
        data = triplets
        results: list[str] = []
        for _, opinion_terms, sentiment in data:
            if sentiment == "netral":
                continue

            doc = self.nlp(opinion_terms)
            if verbose:
                print("Stanza input:", opinion_terms)
                print("Stanza output:")
                print(doc)

            head_word = None
            head_upos = None
            head_id = -1
            for sent in doc.sentences:
                for word in sent.words:
                    if word.head == 0:
                        head_word = word.text
                        head_upos = word.upos
                        head_id = word.id - 1
                        break

                if head_word is not None:
                    break

            assert head_upos is not None
            assert head_id != -1
            modified_opinion_term_words = opinion_terms.split(" ")

            first_head_word = head_word
            while len(modified_opinion_term_words) > 1 and head_upos == "X":
                # We don't want the head to be X UPOS, it's not make sense.
                modified_opinion_term_words.pop(head_id)
                modified_opinion_terms = " ".join(modified_opinion_term_words)
                doc = self.nlp(modified_opinion_terms)
                if verbose:
                    print("Stanza input:", modified_opinion_terms)
                    print("Stanza output:")
                    print(doc)

                head_word = None
                head_upos = None
                head_id = -1
                for sent in doc.sentences:
                    for word in sent.words:
                        if word.head == 0:
                            head_word = word.text
                            head_upos = word.upos
                            head_id = word.id - 1
                            break

                    if head_word is not None:
                        break

                assert head_upos is not None

            if head_upos == "X":
                head_word = first_head_word

            if verbose:
                print("OT to HW:", opinion_terms, "->", head_word)

            assert isinstance(head_word, str)
            results.append(head_word)

        return results

    def _preprocess_text(self, text: str):
        lowercased_text = text.lower()
        preprocessed_text = ""
        current_is_not_a_letter_or_digit = False
        for c in lowercased_text:
            if c in string.ascii_lowercase or c in string.digits:
                if current_is_not_a_letter_or_digit:
                    preprocessed_text += " "
                current_is_not_a_letter_or_digit = False
            else:
                preprocessed_text += " "
                current_is_not_a_letter_or_digit = True

            preprocessed_text += c

        preprocessed_text = " ".join(preprocessed_text.strip().split())
        return preprocessed_text

class StyleDetectorWithAttention(StyleDetectorBase):
    """
    Class for style detector implementation with attention, related to Shi et al.'s (2023) method.
    """

    def __init__(
            self,
            model: RobertaPreTrainedModel,
            tokenizer: RobertaTokenizer,
            layer_index: int,
            head_index: int,
            threshold: float = 0.25,
            word_method: str = "sum",
    ):
        """
        Initialize `StyleDetector` class.

        Args:
        - `model`: RoBERTa model.

        - `tokenizer`: RoBERTa tokenizer.

        - `layer_index`: Chosen layer index (starts from 0).

        - `head_index`: Chosen head index (starts from 0).

        - `threshold`: How many proportion of style words should be chosen for a sentence?

        - `word_method`: `sum` (default) or `max`. How the attention was calculated for a word in many tokens?
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.layer_index = layer_index
        self.head_index = head_index
        self.threshold = threshold
        self.word_method = word_method
        
    def __call__(self, text: str, verbose: bool = False) -> list[str]:
        """
        Detect style words from `text`.

        Args:
        - `text`: Text input.

        - `verbose`: If it's True, print additional informations in the process.
        """
        preped_text = self._preprocess_text(text)
        if verbose:
            print(f"preped_text ({type(preped_text)}):")
            print(preped_text)

        att, decoding_ids = self._get_attention(preped_text)
        if verbose:
            print(f"\natt ({type(att)}):")
            print(att)
            print(f"\ndecoding_ids ({type(decoding_ids)}):")
            print(decoding_ids)

        indices = self._get_filtered_indices(att, decoding_ids)
        if verbose:
            print(f"\nindices ({type(indices)}):")
            print(indices)

        style_words = self._get_filtered_words_by_indices(preped_text, indices)
        return style_words

    def _preprocess_text(self, text: str):
        lowercased_text = text.lower()
        preprocessed_text = ""
        current_is_not_a_letter_or_digit = False
        for c in lowercased_text:
            if c in string.ascii_lowercase or c in string.digits:
                if current_is_not_a_letter_or_digit:
                    preprocessed_text += " "
                current_is_not_a_letter_or_digit = False
            else:
                preprocessed_text += " "
                current_is_not_a_letter_or_digit = True

            preprocessed_text += c

        preprocessed_text = " ".join(preprocessed_text.strip().split())
        return preprocessed_text
    
    def _get_attention(self, text: str):
        """
        This function calculates attention weights of all the heads and
        returns it along with the encoded sentence for further processing.

        text: Text input.
        """

        tokenizer = self.tokenizer # <- Need to implement this one.
        model = self.model
        device = self.device

        ## Preprocessing for RoBERTa Indo
        text_tokens = tokenizer.tokenize(text)
        tokens = ["<s>"] + text_tokens + ["</s>"]
        temp_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(temp_ids)
        segment_id = [0] * len(temp_ids)

        ## Convert the list of int ids to Torch Tensors
        ids = torch.tensor([temp_ids])
        segment_ids = torch.tensor([segment_id])
        input_masks = torch.tensor([input_mask])

        ids = ids.to(device)
        segment_ids = segment_ids.to(device)
        input_masks = input_masks.to(device)

        with torch.no_grad():
            model_result = model(ids, input_masks, segment_ids, output_attentions=True)
            attn = model_result.attentions[self.layer_index]

        attn = attn.to('cpu')
        attn = attn[0]
        attn = attn[self.head_index]

        '''
        attention weights are stored in this way:
        att_lt['length_of_sentence']
        '''
        return attn, ids[0]
    
    def _get_filtered_indices(self, att, decoding_ids):
        """
        This function processes attentions by only taking the top indices with defined threshold value.

        att: attention
        decoding_ids: indexed sentence
        threshold: Percentage of the top indexes to be removed
        word_method: How the attention was chosen for one word in many tokens
        """
        # List of None of num_of_layers * num_of_heads to save the results of each head for input_sentences

        tokenizer = self.tokenizer
        threshold = self.threshold
        word_method = self.word_method

        tok_idx = 1 # Ignore <s>
        token = tokenizer.convert_ids_to_tokens(decoding_ids[tok_idx].item())
        word_att = [att[0][tok_idx]]
        boundary = []

        stop_tok_idx = len(decoding_ids)
        stop = False
        while not stop:
            tok_idx += 1
            if tok_idx == stop_tok_idx:
                print("Warning: end without </s>. This can be lead to error.")
                stop = True
            else:
                token = tokenizer.convert_ids_to_tokens(decoding_ids[tok_idx].item())
                if token == "</s>":
                    stop = True
                elif token.startswith("Ġ"): # Because it just a space separator word.
                    word_att.append(att[0][tok_idx])
                    boundary.append(tok_idx)
                else:
                    if word_method == "sum":
                        word_att[-1] += att[0][tok_idx]
                    elif word_method == "max":
                        word_att[-1] = max(att[0][tok_idx], word_att[-1])
                    else:
                        raise ValueError(f"Unexpected word method: {word_method}")

        boundary.append(tok_idx)

        _, word_topi = torch.Tensor(word_att).topk(len(word_att))
        word_index_list = list(word_topi)
        # Sometimes the taken attention isn't even a word, so we need to blacklist it.

        word_count = 0
        while word_count < len(word_index_list):
            word_idx = word_index_list[word_count]
            start_tok_idx = 1 if word_idx == 0 else boundary[word_idx - 1]
            end_tok_idx = boundary[word_idx]

            found = False
            for tok_idx in range(start_tok_idx, end_tok_idx):
                token = tokenizer.convert_ids_to_tokens(decoding_ids[tok_idx].item())
                if any(c in token for c in "abcdefghijklmnopqrstuvwxyz"):
                    found = True
                    break

            if found:
                word_count += 1
            else:
                word_index_list.pop(word_count)

        # word_index_list may be modified, word_count == len(word_index_list)
        word_count = 0
        while word_count < len(word_index_list) * threshold:
            word_count += 1

        # word_count >= len(word_index_list) * threshold
        word_topi = [x.item() for x in word_index_list[:word_count]]
        word_topi.sort()
        processed_indices = word_topi

        return processed_indices
    
    def _get_filtered_words_by_indices(self, text: str, indices: list[int]):
        words = text.split(" ")
        filtered_words = [w for j, w in enumerate(words) if j in indices]
        return filtered_words
