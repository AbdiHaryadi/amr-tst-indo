import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")

import fasttext
from nltk.corpus import wordnet as wn
import penman
import re
from transformers import pipeline, TextClassificationPipeline

class StyleRewriting:
    """
    Class for style rewriting implementation, based on [Shi et al. (2023)](https://aclanthology.org/2023.findings-acl.260.pdf).
    """

    def __init__(
            self,
            clf_pipeline: TextClassificationPipeline | str,
            fasttext_model_path: str,
            *,
            lang: str = "ind",
            word_expand_size: int = 10,
            ignore_and_warn_if_target_word_not_found: bool = True,
            max_score_strategy: bool = False,
            remove_polarity_strategy: bool = True,
            reset_sense_strategy: bool = True
    ):
        """
        Initialize `StyleRewriting` class.

        Args:
        - `clf_pipeline`: Pipeline for style classification. If it's a string,
        it should be a Huggingface checkpoint for style classifier. The
        checkpoint should support `text-classification` pipeline.

        - `fasttext_model_path`: The Fasttext model path, trained from
        [Fasttext library](https://fasttext.cc/).

        - `word_expand_size`: Number of words expanded from Fasttext in the case
        of no antonym from WordNet. The default value is 10.

        - `lang`: Supported language. Indonesia (`ind`) is the default.

        - `ignore_and_warn_if_target word_not_found`: Default to `True`.

        - `max_score_strategy`: Default to `False`. If it's `True`, for the case
        of multiple valid target words, instead of returning the first word like
        in the original paper, the classifier takes into account so the returned
        word has the maximum score.

        - `remove_polarity_strategy`: Default to `True`. If at least one involved
        node has negative polarity for specific polarity, remove that polarity
        instead of finding the antonym. If there is no polarity, don't change.

        - `reset_sense_strategy`: Default to `True`. For AMR frames which are
        consisted to style words, it will rewrite to the new AMR frame with
        `-00` sense number. If the value is `False`, the target sense number is
        same as source sense number.
        """

        if isinstance(clf_pipeline, str):
            self.clf_pipe = pipeline("text-classification", model=clf_pipeline)
        else:
            self.clf_pipe = clf_pipeline
        
        self.fasttext_model = fasttext.load_model(fasttext_model_path)
        self.word_expand_size = word_expand_size
        self.lang = lang
        self.ignore_and_warn_if_target_word_not_found = ignore_and_warn_if_target_word_not_found
        self.max_score_strategy = max_score_strategy
        self.remove_polarity_strategy = remove_polarity_strategy
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

            if self.remove_polarity_strategy and self._is_word_consistent_with_negative_polarity_node_in_amr(new_amr, w):
                new_amr = self._remove_polarity(new_amr, w)
            
            elif self._is_word_consistent_with_node_in_amr(new_amr, w):
                target_w = self._get_target_style_word(source_style, w, text_without_style_words, verbose)
                if target_w is None:
                    if self.ignore_and_warn_if_target_word_not_found:
                        print(f"Warning: For text \"{text}\", target word for \"{w}\" is not found with source style {source_style}. Ignored.")
                    else:
                        raise NotImplementedError(f"For text \"{text}\", target word for \"{w}\" is not found with source style {source_style}.")
                else:
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

            for current_style_word in style_word_list:
                current_antonym_list = self._get_antonym_list(current_style_word)
                if verbose:
                    print(f"{current_style_word=}")
                    print(f"{current_antonym_list=}")

                for a in current_antonym_list:
                    if a not in antonym_list:
                        antonym_list.append(a)

                tried_style_word_set.add(current_style_word)

            if verbose:
                print(f"{style_word_list=}")
                print(f"{antonym_list=}")
                
            max_score = 0.0
            chosen_a = None
            for a in antonym_list:
                x_tmp = text_without_style_words + " " + a
                try:
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
                    
                except Exception as e:
                    print(f"Error when processing \"{x_tmp}\"!\nError: {e}\nBecause of that, {a} is chosen as valid target style word.")
                    chosen_a = a
                    break

            # All antonym_list element has been iterated
            if chosen_a is not None:
                return chosen_a
            # else: expand the words

            new_style_word_list = []
            for current_style_word in style_word_list:
                fasttext_result = self.fasttext_model.get_nearest_neighbors(current_style_word, k=self.word_expand_size)
                if verbose:
                    print(f"{current_style_word=}")
                    print(f"{fasttext_result=}")

                for _, new_style_word in fasttext_result:
                    if new_style_word not in tried_style_word_set:
                        new_style_word_list.append(new_style_word)

            if verbose:
                print(f"{tried_style_word_set=}")
                print(f"{new_style_word_list=}")

            style_word_list = new_style_word_list

        return None

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
    
    def _is_word_consistent_with_negative_polarity_node_in_amr(self, amr: penman.Graph, word: str):
        for var, rel, instance in amr.triples:
            if rel == ":instance" and self._is_word_consistent_with_instance(word, instance):
                for other_var, other_rel, polarity in amr.triples:
                    if other_var == var and other_rel == ":polarity" and polarity == "-":
                        return True

        return False

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

    def _remove_polarity(self, amr: penman.Graph, source_word: str):
        new_amr = amr
        for var, rel, instance in amr.triples:
            if rel == ":instance":
                if self._is_word_consistent_with_instance(source_word, instance):
                    new_amr = self._remove_polarity_at_var_if_exists(new_amr, var)

        return new_amr
    
    def _remove_polarity_at_var_if_exists(self, amr: penman.Graph, selected_var: str):
        new_triples = []
        new_epidata = {}

        # Find polarity
        found = False
        prev_t = None
        for t, op in amr.epidata.items():
            if not found:
                var, rel, polarity = t
                if var == selected_var and rel == ":polarity" and polarity == "-":
                    # Remove this.
                    assert prev_t is not None
                    new_epidata[prev_t] += op
                    found = True
                    continue

            new_triples.append(t)
            new_epidata[t] = op
            prev_t = t
        
        return penman.Graph(triples=new_triples, top=amr.top, epidata=new_epidata, metadata=amr.metadata)
    
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
                    new_instance = new_instance.replace("_", " ")
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

class TextBasedStyleRewriting:
    """
    Class like `StyleRewriting`, but with text only; no AMR required.
    """

    def __init__(
            self,
            clf_pipeline: TextClassificationPipeline | str,
            fasttext_model_path: str,
            *,
            lang: str = "ind",
            word_expand_size: int = 10,
            ignore_and_warn_if_target_word_not_found: bool = True,
            max_score_strategy: bool = False
    ):
        """
        Initialize `TextBasedStyleRewriting` class.

        Args:
        - `clf_pipeline`: Pipeline for style classification. If it's a string,
        it should be a Huggingface checkpoint for style classifier. The
        checkpoint should support `text-classification` pipeline.

        - `fasttext_model_path`: The Fasttext model path, trained from
        [Fasttext library](https://fasttext.cc/).

        - `word_expand_size`: Number of words expanded from Fasttext in the case
        of no antonym from WordNet. The default value is 10.

        - `lang`: Supported language. Indonesia (`ind`) is the default.

        - `ignore_and_warn_if_target word_not_found`: Default to `True`.

        - `max_score_strategy`: Default to `False`. If it's `True`, for the case
        of multiple valid target words, instead of returning the first word like
        in the original paper, the classifier takes into account so the returned
        word has the maximum score.
        """

        if isinstance(clf_pipeline, str):
            self.clf_pipe = pipeline("text-classification", model=clf_pipeline)
        else:
            self.clf_pipe = clf_pipeline
        
        self.fasttext_model = fasttext.load_model(fasttext_model_path)
        self.word_expand_size = word_expand_size
        self.lang = lang
        self.ignore_and_warn_if_target_word_not_found = ignore_and_warn_if_target_word_not_found
        self.max_score_strategy = max_score_strategy

        self.last_log = []

    def __call__(
            self,
            text: str,
            source_style: str,
            style_words: list[str],
            verbose: bool = False
    ):
        """
        Rewrite text based on source style and style words.

        Args:
        - `text`: The involved text which used for generating `amr`.

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

        self.last_log = []

        text_without_style_words = self._get_text_without_style_words(text, style_words)

        new_text = text
        handled_style_words = []

        for w in style_words:
            if w in handled_style_words:
                continue

            if w in new_text:
                target_w = self._get_target_style_word(source_style, w, text_without_style_words, verbose)
                if target_w is None:
                    if self.ignore_and_warn_if_target_word_not_found:
                        print(f"Warning: For text \"{text}\", target word for \"{w}\" is not found with source style {source_style}. Ignored.")
                        self.last_log.append({
                            "type": "target_word_not_found",
                            "text": text,
                            "source_style": source_style,
                            "word": w
                        })
                    else:
                        raise NotImplementedError(f"For text \"{text}\", target word for \"{w}\" is not found with source style {source_style}.")
                else:
                    new_text = self._rewrite_text(new_text, w, target_w)

            handled_style_words.append(w)
            
        return new_text

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

            for current_style_word in style_word_list:
                current_antonym_list = self._get_antonym_list(current_style_word)
                if verbose:
                    print(f"{current_style_word=}")
                    print(f"{current_antonym_list=}")
                self.last_log.append({
                    "type": "get_antonyms",
                    "word": current_style_word,
                    "antonyms": current_antonym_list
                })

                for a in current_antonym_list:
                    modified_a = a.replace("_", " ")
                    if modified_a not in antonym_list:
                        antonym_list.append(modified_a)

                tried_style_word_set.add(current_style_word)

            if verbose:
                print(f"{style_word_list=}")
                print(f"{antonym_list=}")
                
            max_score = 0.0
            chosen_a = None
            for a in antonym_list:
                x_tmp = text_without_style_words + " " + a
                try:
                    pipe_result, *_ = self.clf_pipe(x_tmp)
                    if verbose:
                        print(f"{x_tmp=}")
                        print(f"{pipe_result=}")
                    self.last_log.append({
                        "type": "check_style",
                        "text": x_tmp
                    } | pipe_result)

                    if pipe_result["label"] != source_style:
                        if (not self.max_score_strategy) or (max_score < pipe_result["score"]):
                            chosen_a = a
                            if not self.max_score_strategy:
                                break

                            max_score = pipe_result["score"]
                    
                except Exception as e:
                    print(f"Error when processing \"{x_tmp}\"!\nError: {e}\nBecause of that, {a} is chosen as valid target style word.")
                    chosen_a = a
                    break

            # All antonym_list element has been iterated
            if chosen_a is not None:
                return chosen_a
            # else: expand the words

            new_style_word_list = []
            for current_style_word in style_word_list:
                fasttext_result = self.fasttext_model.get_nearest_neighbors(current_style_word, k=self.word_expand_size)
                if verbose:
                    print(f"{current_style_word=}")
                    print(f"{fasttext_result=}")

                for _, new_style_word in fasttext_result:
                    if new_style_word not in tried_style_word_set:
                        new_style_word_list.append(new_style_word)

            if verbose:
                print(f"{tried_style_word_set=}")
                print(f"{new_style_word_list=}")

            self.last_log.append({
                "type": "expand",
                "old_words": style_word_list,
                "new_words": new_style_word_list
            })

            style_word_list = new_style_word_list

        return None

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
    
    def _rewrite_text(self, text: str, source_word: str, target_word: str):
        if any(c in source_word for c in r"[]\.^$*+?{}|()"):
            print(f"Warning: Source word ({source_word}) contains regex metacharacter. It may rewrite text incorrectly!")

        return re.sub(rf"\b({source_word})\b", target_word, text)
    
    def get_last_log(self):
        return self.last_log
