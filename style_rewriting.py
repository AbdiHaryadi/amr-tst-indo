import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")

import fasttext
from nltk.corpus import wordnet as wn
import penman
import re
from transformers import pipeline

class AMRStyleRewriting:
    """
    Class for rewriting AMR from specific style, based on [Shi et al. (2023)](https://aclanthology.org/2023.findings-acl.260.pdf).
    """

    def __init__(
            self,
            style_clf_hf_checkpoint: str,
            fasttext_model_path: str,
            lang: str = "ind",
            word_expand_size: int = 10,
            max_score_strategy: bool = False,
            reset_sense_strategy: bool = True
    ):
        """
        Initialize `AMRStyleRewriting` class.

        Args:
        - `style_clf_hf_checkpoint`: The Huggingface checkpoint for style
        classifier. The checkpoint should support `text-classification`
        pipeline.

        - `fasttext_model_path`: The Fasttext model path, trained from
        [Fasttext library](https://fasttext.cc/).

        - `word_expand_size`: Number of words expanded from Fasttext in the case
        of no antonym from WordNet. The default value is 10.

        - `lang`: Supported language. Indonesia (`ind`) is the default.

        - `max_score_strategy`: Default to `False`. If it's `True`, for the case
        of multiple valid target words, instead of returning the first word like
        in the original paper, the classifier takes into account so the returned
        word has the maximum score.

        - `reset_sense_strategy`: Default to `True`. For AMR frames which are
        consisted to style words, it will rewrite to the new AMR frame with
        `-00` sense number. If the value is `False`, the target sense number is
        same as source sense number.
        """

        self.clf_pipe = pipeline("text-classification", model=style_clf_hf_checkpoint)
        self.fasttext_model = fasttext.load_model(fasttext_model_path)
        self.word_expand_size = word_expand_size
        self.lang = lang
        self.max_score_strategy = max_score_strategy
        self.reset_sense_strategy = reset_sense_strategy

    def __call__(
            self,
            text: str,
            amr: penman.Graph,
            source_style: str,
            style_words: list[str],
            verbose: bool = False
    ):
        """
        Rewrite AMR based on text, source style, and style words.

        Args:
        - `text`: The involved text which used for generating `amr`.

        - `amr`: The AMR graph.

        - `source_style`: The source style of text. It is dependent to
        `style_clf_hf_checkpoint`.
        For example, in
        [this model](abdiharyadi/roberta-base-indonesian-522M-with-sa-william-dataset),
        negative sentiment is called `LABEL_0`, and positive sentiment is called
        `LABEL_1`.

        - `style_words`: The identified style words from `text`.

        - `verbose`: Print step information for debugging purpose. Default to
        `False`.
        """

        text_without_style_words = self._get_text_without_style_words(text, style_words)

        new_amr = amr
        handled_style_words = []

        for w in style_words:
            if w in handled_style_words:
                continue
            
            if self._is_word_consistent_with_node_in_amr(new_amr, w):
                target_w = self._get_target_style_word(source_style, w, text_without_style_words, verbose)
                new_amr = self._rewrite_amr_nodes(new_amr, w, target_w)

            handled_style_words.append(w)
            
        return new_amr

    def _get_antonym_list(self, word: str):
        main_synsets = wn.synsets(word, lang=self.lang)
        main_lemma_list = []
        for ss in main_synsets:
            for l in ss.lemmas():
                if l in main_lemma_list:
                    continue

                main_lemma_list.append(l)

        antonym_list = []
        for l in main_lemma_list:
            for a in l.antonyms():
                for ln in a.synset().lemma_names(lang=self.lang):
                    if ln in antonym_list:
                        continue

                    antonym_list.append(ln)

        return antonym_list

    def _get_target_style_word(self, source_style: str, style_word: str, text_without_style_words: str, verbose: bool):
        tried_style_word_set = set()
        style_word_list = [style_word]
        while len(style_word_list) > 0:
            antonym_list = []

            for style_word in style_word_list:
                current_antonym_list = self._get_antonym_list(style_word)
                if verbose:
                    print(f"{style_word=}")
                    print(f"{current_antonym_list=}")

                for a in current_antonym_list:
                    if a not in antonym_list:
                        antonym_list.append(a)

                tried_style_word_set.add(style_word)

            if verbose:
                print(f"{style_word_list=}")
                print(f"{antonym_list=}")
                
            max_score = 0.0
            chosen_a = None
            for a in antonym_list:
                x_tmp = text_without_style_words + " " + a
                pipe_result, *_ = self.clf_pipe(x_tmp)
                if verbose:
                    print(f"{x_tmp=}")
                    print(f"{pipe_result=}")

                if pipe_result["label"] != source_style:
                    if (not self.max_score_strategy) or (max_score < pipe_result["score"]):
                        chosen_a = a
                        if not self.max_score_strategy:
                            break

                        max_score = pipe_result["score"]

            # All antonym_list element has been iterated
            if chosen_a is not None:
                return chosen_a

            new_style_word_list = []
            for style_word in style_word_list:
                fasttext_result = self.fasttext_model.get_nearest_neighbors(style_word, k=self.word_expand_size)
                if verbose:
                    print(f"{style_word=}")
                    print(f"{fasttext_result=}")

                for _, new_style_word in fasttext_result:
                    if new_style_word not in tried_style_word_set:
                        new_style_word_list.append(new_style_word)

            if verbose:
                print(f"{tried_style_word_set=}")
                print(f"{new_style_word_list=}")

            style_word_list = new_style_word_list

        raise NotImplementedError("Cannot find antonym style words, even with Fasttext, and no implementation for that case.")

    def _get_text_without_style_words(self, text: str, style_words: list[str]):
        ss_tokens = text.split(" ")
        new_ss_tokens = []

        for t in ss_tokens:
            new_t = t
            for w in style_words:
                new_t = new_t.replace(w, "")
            
            new_t = new_t.strip()
            if new_t != "":
                new_ss_tokens.append(new_t)

        return " ".join(new_ss_tokens)

    def _is_frame(self, instance: str):
        """
        Frame format: lemma-lemmi-XX, X is number digit
        """
        return re.match(r"[a-z]+(-[a-z])*-\d\d", instance)

    def _is_word_consistent_with_node_in_amr(self, amr: penman.Graph, word: str):
        for instance in amr.instances():
            if self._is_word_consistent_with_instance(word, instance.target):
                return True

        return False

    def _is_word_consistent_with_instance(self, word: str, instance: str):
        mod_word = word.replace(" ", "-")
        if self._is_frame(instance):
            if mod_word == instance[:-3]:
                return True
        else:
            if mod_word == instance:
                return True

        return False

    def _rewrite_amr_nodes(self, amr: penman.Graph, source_word: str, target_word: str):
        new_amr = amr
        for var, rel, instance in amr.triples:
            if rel == ":instance":
                if self._is_word_consistent_with_instance(source_word, instance):
                    new_amr = self._rewrite_amr_node_at_var(new_amr, source_word, var, target_word)

        return new_amr

    def _rewrite_amr_node_at_var(self, amr: penman.Graph, source_word: str, selected_var: str, target_word: str):
        new_triples = []
        new_epidata = {}

        found = False
        for t, op in amr.epidata.items():
            if not found:
                var, rel, instance = t
                if var == selected_var and rel == ":instance":
                    new_instance = instance.replace("-", " ")
                    new_instance = new_instance.replace(source_word, target_word)
                    new_instance = new_instance.replace(" ", "-")
                    if self.reset_sense_strategy and self._is_frame(new_instance):
                        new_instance = new_instance[:-3] + "-00"

                    new_t = (var, rel, new_instance)
                    new_triples.append(new_t)
                    new_epidata[new_t] = op

                    found = True
                    continue

            new_triples.append(t)
            new_epidata[t] = op

        if not found:
            raise ValueError(f"Cannot found an instance from \"{selected_var}\"")

        return penman.Graph(triples=new_triples, top=amr.top, epidata=new_epidata, metadata=amr.metadata)
