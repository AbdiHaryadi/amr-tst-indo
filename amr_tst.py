import penman
from amr_to_text import AMRToText
from dataclasses import dataclass
from style_detector import StyleDetector
from style_rewriting import StyleRewriting
from text_to_amr import TextToAMR
from utils import make_no_metadata_graph

@dataclass
class AMRTSTDetailedResult:
    source_text: list[str]
    source_style: list[str]
    source_amr: list[penman.Graph]
    style_words: list[list[str]]
    target_amr: list[penman.Graph]
    target_text: list[str]

    def to_list(self) -> list[dict[str, str | list[str]]]:
        expected_length = len(expected_length)
        assert (other_length := len(self.source_style)) == expected_length, f"Expecting {expected_length}, got {other_length}"
        assert (other_length := len(self.source_amr)) == expected_length, f"Expecting {expected_length}, got {other_length}"
        assert (other_length := len(self.style_words)) == expected_length, f"Expecting {expected_length}, got {other_length}"
        assert (other_length := len(self.target_amr)) == expected_length, f"Expecting {expected_length}, got {other_length}"
        assert (other_length := len(self.target_text)) == expected_length, f"Expecting {expected_length}, got {other_length}"

        data: list[dict[str, str | list[str]]] = []
        for i in range(expected_length):
            sa = make_no_metadata_graph(self.source_amr[i])
            ta = make_no_metadata_graph(self.target_amr[i])
            
            data.append({
                "source_text": self.source_text[i],
                "source_style": self.source_style[i],
                "source_amr": penman.encode(sa, indent=None),
                "style_words": self.style_words[i],
                "target_amr": penman.encode(ta, indent=None),
                "target_text": self.target_text[i],
            })

        return data

class AMRTST:
    """
    Class for AMR-TST implementation, based on [Shi et al. (2023)](https://aclanthology.org/2023.findings-acl.260.pdf).
    """

    def __init__(
            self,
            t2a: TextToAMR,
            sd: StyleDetector,
            sr: StyleRewriting,
            a2t: AMRToText
    ):
        self.t2a = t2a
        self.sd = sd
        self.sr = sr
        self.a2t = a2t

    def __call__(self, texts: list[str], source_styles: list[str]):
        assert len(texts) == len(source_styles)
        graphs = self.t2a(texts)

        style_words_list: list[list[str]] = []
        rewritten_graphs: list[penman.Graph] = []

        for t, g, s in zip(texts, graphs, source_styles):
            current_style_words = self.sd(t)
            style_words_list.append(current_style_words)

            g_tgt = self.sr(t, g, s, current_style_words)
            rewritten_graphs.append(g_tgt)

        target_texts = self.a2t(rewritten_graphs)
        return target_texts, AMRTSTDetailedResult(
            source_text=texts,
            source_style=source_styles,
            source_amr=graphs,
            style_words=style_words_list,
            target_amr=rewritten_graphs,
            target_text=target_texts
        )
