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
        print("Warning: because AMR Generator is not implemented yet, the return is graphs.")
        graphs = self.t2a(texts)

        target_graphs: list[penman.Graph] = []
        for t, g, s in zip(texts, graphs, source_styles):
            current_style_words = self.sd(t)
            g_tgt = self.sr(t, g, s, current_style_words)
            target_graphs.append(g_tgt)

        return target_graphs

    @staticmethod
    def load(
            t2a_model_name: str,
            sd_config_path: str,
            sd_model_path: str,
            sr_style_clf_hf_checkpoint: str,
            sr_fasttext_model_path: str,
            *,
            t2a_hf_repo_id: str | None = "",
            t2a_hf_kwargs: dict = {},
            t2a_kwargs: dict = {},
            sd_kwargs: dict = {},
            sr_kwargs: dict = {}
    ):
        if t2a_hf_repo_id is None:
            t2a = TextToAMR(model_name=t2a_model_name, **t2a_kwargs)
        else:
            t2a = TextToAMR.from_huggingface(
                t2a_hf_repo_id,
                model_name=t2a_model_name,
                hf_kwargs=t2a_hf_kwargs,
                **t2a_kwargs
            )
        
        sd = StyleDetector(
            config_path=sd_config_path,
            model_path=sd_model_path,
            **sd_kwargs
        )
        sr = StyleRewriting(
            style_clf_hf_checkpoint=sr_style_clf_hf_checkpoint,
            fasttext_model_path=sr_fasttext_model_path,
            **sr_kwargs
        )

        return AMRTST(t2a=t2a, sd=sd, sr=sr)
