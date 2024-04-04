# Postprocess Interface to check prediction and make normalization to non-correct aspect/sentiment term...
# Current Implementation: EditDistancePostProcessor, EmbeddingDistancePostProcessor
# Postprocess Interface:
from typing import Tuple, List
import string

from src.constant import GENERAL_ASPECT, GENERAL_ASPECTS

sentiment_word_list = ["positif", "negatif", "netral"]

special_chars = r'"!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~"'

class IPostprocess:
    # == Generalize strategy ==
    def check_and_fix_preds(
        self, all_pairs: List[List[Tuple[str, str, str]]], sents: List[List[str]], implicit=False
    ) -> List[List[Tuple[str, str, str]]]:
        
        all_new_pairs = []

        for i, pairs in enumerate(all_pairs):
            new_pairs = []
            # -- [Handling failed to parse by system] --
            if pairs == []:
                all_new_pairs.append(pairs)
            else:
                for pair in pairs:
                    at, ot, polarity = pair

                    # --- [Recovering aspect term] ---
                    at = at.translate(str.maketrans('', '', special_chars)) # Remove Special character...
                    if at not in sents[i]:
                        if implicit and at in GENERAL_ASPECTS:
                            new_at = GENERAL_ASPECT
                        else:
                            new_at = self.recover_term(at, sents[i])
                    else:
                        new_at = at

                    # --- [Recovering polarity term] ---
                    if polarity not in sentiment_word_list:
                        new_sentiment = self.recover_term(polarity, sentiment_word_list)
                    else:
                        new_sentiment = polarity

                    # -- [Recovering opinion term]
                    if ot not in sents[i]:
                        new_ot = self.recover_term(ot, sents[i])
                    else:
                        new_ot = ot

                    new_pairs.append((new_at, new_ot, new_sentiment))
                all_new_pairs.append(new_pairs)
        return all_new_pairs

    def recover_term(self, original_term: str, sent: List[str]) -> str:
        pass
