import sys
sys.path.append("./indonesian-aste-generative")

from demo import load_generator
import stanza

class StyleDetector:
    """
    Class for style detector implementation, based on [Shi et al. (2023)](https://aclanthology.org/2023.findings-acl.260.pdf).
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

    def __call__(self, text: str, verbose: bool = False) -> list[str]:
        """
        Detect style words from `text`.

        Args:
        - `text`: Pretokenized text. The token should be space-separated.

        - `verbose`: If it's True, print additional informations in the process.
        """
        data = self.generator(text)
        if verbose:
            for x in data:
                print("Triplet:", x)
        
        results: list[str] = []
        for _, opinion_terms, sentiment in data:
            if sentiment == "netral":
                continue

            doc = self.nlp(opinion_terms)
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
                doc = self.nlp(" ".join(modified_opinion_term_words))
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
