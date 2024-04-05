import editdistance
from typing import List

from src.constant import GENERAL_ASPECT
from src.postprocess.interface import IPostprocess, special_chars

class EditDistancePostProcessor(IPostprocess):
    # == Levenshtein distance normalization strategy ==
    def recover_term(self, original_term: str, sent: List[str]) -> str:
        if original_term == GENERAL_ASPECT:
            return original_term

        words = original_term.split(" ")
        new_words = []
        for word in words:
            edit_dis = []
            for token in sent:
                if token not in special_chars:
                    edit_dis.append(editdistance.eval(word, token))
                else:
                    edit_dis.append(float("inf"))
            smallest_idx = edit_dis.index(min(edit_dis))
            new_words.append(sent[smallest_idx])
        new_term = " ".join(new_words)
        return new_term
