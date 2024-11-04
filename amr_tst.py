import penman
from amr_to_text import AMRToTextBase
from dataclasses import dataclass
from style_detector import StyleDetector
from style_rewriting import StyleRewriting
from text_to_amr import TextToAMR
from utils import make_no_metadata_graph
from tqdm import tqdm
from typing import Optional

BACKOFF = penman.Graph(
    [
        penman.Triple("d2", ":instance", "dog"),
        penman.Triple("b1", ":instance", "bark-01"),
        penman.Triple("b1", ":ARG0", "d2"),
    ]
)

@dataclass
class AMRTSTDetailedResult:
    source_text: list[str]
    source_style: list[str]
    source_amr: list[penman.Graph]
    style_words: list[list[str]]
    target_amr: list[penman.Graph]
    target_text: list[str]
    style_rewriting_log: list[dict[str]]

    def to_list(self) -> list[dict[str, str | list[str] | dict[str]]]:
        expected_length = len(self.source_text)
        assert (other_length := len(self.source_style)) == expected_length, f"Expecting {expected_length}, got {other_length}"
        assert (other_length := len(self.source_amr)) == expected_length, f"Expecting {expected_length}, got {other_length}"
        assert (other_length := len(self.style_words)) == expected_length, f"Expecting {expected_length}, got {other_length}"
        assert (other_length := len(self.target_amr)) == expected_length, f"Expecting {expected_length}, got {other_length}"
        assert (other_length := len(self.target_text)) == expected_length, f"Expecting {expected_length}, got {other_length}"
        assert (other_length := len(self.style_rewriting_log)) == expected_length, f"Expecting {expected_length}, got {other_length}"

        data: list[dict[str, str | list[str] | dict[str]]] = []
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
                "style_rewriting_log": self.style_rewriting_log[i],
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
            a2t: AMRToTextBase
    ):
        self.t2a = t2a
        self.sd = sd
        self.sr = sr
        self.a2t = a2t

        self.last_source_graphs = []
        self.last_style_words_list = []
        self.last_rewritten_graphs = []
        self.last_target_texts = []

    def __call__(
            self,
            texts: list[str],
            source_styles: list[str],
            precomputed_graphs: Optional[list[penman.Graph]] = None,
            precomputed_style_words: Optional[list[list[str]]] = None
    ):
        number_of_data = len(texts)
        assert len(source_styles) == number_of_data
        if precomputed_graphs is None:
            graphs = self.t2a(texts)
        else:
            assert len(precomputed_graphs) == number_of_data
            graphs = precomputed_graphs
        self.last_source_graphs = graphs

        if precomputed_style_words is None:
            self.last_style_words_list = []
            style_words_list: list[list[str]] = self.last_style_words_list

            for t, g, s in tqdm(zip(texts, graphs, source_styles), total=number_of_data):
                try:
                    current_style_words = self.sd(t)
                except Exception as e:
                    print(f"Warning: For text {t}, style detector cannot be executed.\nError: {e}")
                    current_style_words = []
                style_words_list.append(current_style_words)
            
        else:
            self.last_style_words_list = precomputed_style_words
            style_words_list = precomputed_style_words
        
        self.last_rewritten_graphs = []
        rewritten_graphs: list[penman.Graph] = self.last_rewritten_graphs
        rewrite_logs: list[dict[str]] = []

        for t, g, s, current_style_words in tqdm(zip(texts, graphs, source_styles, style_words_list), total=number_of_data):
            try:
                g_tgt = self.sr(t, g, s, current_style_words)
            except Exception as e:
                print(f"Warning: For text {t}, graph {penman.encode(g)}, style {s}, and style words {current_style_words}, style rewriting cannot be processed.\nError: {e}")
                g_tgt = BACKOFF
            
            rewrite_logs.append(self.sr.get_last_log())
            rewritten_graphs.append(g_tgt)

        try:
            target_texts = self.a2t(rewritten_graphs)
        except Exception as e:
            print(f"Warning: Can't process all of the target graphs for AMR generation.\nError: {e}")
            target_texts = ["" for _ in rewritten_graphs]
        self.last_target_texts = target_texts

        return target_texts, AMRTSTDetailedResult(
            source_text=texts,
            source_style=source_styles,
            source_amr=graphs,
            style_words=style_words_list,
            target_amr=rewritten_graphs,
            target_text=target_texts,
            style_rewriting_log=rewrite_logs
        )
