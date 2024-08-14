from amr_to_text import AMRToText
from amr_tst import AMRTST
import penman
from style_detector import StyleDetector
from style_rewriting import StyleRewriting
from text_to_amr import TextToAMR

def test_when_using_precomputed_style_words():
    class DummyTextToAMR(TextToAMR):
        def __init__(self):
            pass

        def __call__(self, texts: list[str]):
            raise NotImplementedError("TextToAMR shouldn't be used!")
        
    class DummyStyleDetector(StyleDetector):
        def __init__(self):
            pass

        def __call__(self, text: str):
            raise NotImplementedError("StyleDetector shouldn't be used!")
        
    class DummyStyleRewriting(StyleRewriting):
        def __init__(self):
            pass

        def __call__(
                self,
                text: str,
                amr: penman.Graph,
                source_style: str,
                style_words: list[str]
        ):
            dummy_string = "".join(style_words)
            return penman.decode(f"(z0 / string-entity :value \"list{dummy_string}\" )")
    
    class DummyAMRToText(AMRToText):
        def __init__(self):
            pass

        def __call__(self, graphs: list[penman.Graph]):
            return [penman.encode(g, indent=None) for g in graphs]
        
    amr_tst = AMRTST(
        t2a=DummyTextToAMR(),
        sd=DummyStyleDetector(),
        sr=DummyStyleRewriting(),
        a2t=DummyAMRToText(),
    )

    texts = [
        "kalimat1",
        "kalimat2",
        "kalimat3"
    ]
    source_styles = ["idc" for _ in texts]
    precomputed_graphs = [penman.decode("(z0 / string-entity :value \"idc\" )") for _ in texts]
    precomputed_style_words = [
        ["katapertama"],
        [],
        ["katakedua", "kataketiga"]
    ]

    results, _ = amr_tst(
        texts=texts,
        source_styles=source_styles,
        precomputed_graphs=precomputed_graphs,
        precomputed_style_words=precomputed_style_words
    )

    assert results[0] == penman.encode(
        penman.decode("(z0 / string-entity :value \"listkatapertama\" )"),
        indent=None
    )
    assert results[1] == penman.encode(
        penman.decode("(z0 / string-entity :value \"list\" )"),
        indent=None
    )
    assert results[2] == penman.encode(
        penman.decode("(z0 / string-entity :value \"listkatakeduakataketiga\" )"),
        indent=None
    )
    assert len(results) == 3

def test_get_style_rewriting_logs():
    class DummyTextToAMR(TextToAMR):
        def __init__(self):
            pass

        def __call__(self, texts: list[str]):
            raise NotImplementedError("TextToAMR shouldn't be used!")
        
    class DummyStyleDetector(StyleDetector):
        def __init__(self):
            pass

        def __call__(self, text: str):
            raise NotImplementedError("StyleDetector shouldn't be used!")
        
    class DummyStyleRewriting(StyleRewriting):
        def __init__(self):
            pass

        def __call__(
                self,
                text: str,
                amr: penman.Graph,
                source_style: str,
                style_words: list[str]
        ):
            self.last_log = style_words
            return penman.decode(f"(z0 / string-entity :value \"idc2\" )")
    
    class DummyAMRToText(AMRToText):
        def __init__(self):
            pass

        def __call__(self, graphs: list[penman.Graph]):
            return [penman.encode(g, indent=None) for g in graphs]
        
    amr_tst = AMRTST(
        t2a=DummyTextToAMR(),
        sd=DummyStyleDetector(),
        sr=DummyStyleRewriting(),
        a2t=DummyAMRToText(),
    )

    texts = [
        "kalimat1",
        "kalimat2",
        "kalimat3"
    ]
    source_styles = ["idc" for _ in texts]
    precomputed_graphs = [penman.decode("(z0 / string-entity :value \"idc\" )") for _ in texts]
    precomputed_style_words = [
        ["katapertama"],
        [],
        ["katakedua", "kataketiga"]
    ]

    _, details = amr_tst(
        texts=texts,
        source_styles=source_styles,
        precomputed_graphs=precomputed_graphs,
        precomputed_style_words=precomputed_style_words
    )
    info_list = details.to_list()
    assert info_list[0]["style_rewriting_log"] == precomputed_style_words[0]
    assert info_list[1]["style_rewriting_log"] == precomputed_style_words[1]
    assert info_list[2]["style_rewriting_log"] == precomputed_style_words[2]
    assert len(info_list) == 3
