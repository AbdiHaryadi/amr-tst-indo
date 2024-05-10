import penman
from style_detector import StyleDetector
from style_rewriting import StyleRewriting
from text_to_amr import TextToAMR

class AMRTST:
    """
    Class for AMR-TST implementation, based on [Shi et al. (2023)](https://aclanthology.org/2023.findings-acl.260.pdf).
    """

    def __init__(
            self,
            t2a: TextToAMR,
            sd: StyleDetector,
            sr: StyleRewriting
    ):
        self.t2a = t2a
        self.sd = sd
        self.sr = sr

    def __call__(self, texts: list[str], source_styles: list[str]):
        assert len(texts) == len(source_styles)

        print("Warning: AMR Generator is not implemented, returning (list of empty string, infos)")
        graphs = self.t2a(texts)

        infos: dict[list] = {
            "source_amr": [penman.encode(g) for g in graphs],
            "style_words": [],
            "target_amr": []
        }

        target_texts: list[str] = []

        for t, g, s in zip(texts, graphs, source_styles):
            current_style_words = self.sd(t)
            infos["style_words"].append(current_style_words)

            g_tgt = self.sr(t, g, s, current_style_words)
            infos["target_amr"].append(penman.encode(g_tgt))    

            # TODO: AMR Generator here.
            target_texts.append("")
        
        return target_texts, infos
