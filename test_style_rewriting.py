from style_rewriting import StyleRewriting
import penman

def test_handle_target_word_with_underscore():
    class DummyStyleRewriting(StyleRewriting):
        def __init__(self):
            self.remove_polarity_strategy = True
            self.max_score_strategy = False
            self.reset_sense_strategy = True
            self.clf_pipe = lambda x: [{"label": "LABEL_1"}]

        def _get_antonym_list(self, word: str):
            if word == "sedikit":
                return ["besar_hati"]
            else:
                raise NotImplementedError

    sr = DummyStyleRewriting()
    text = "saya meminta thai hot dan mendapatkan rempah yang sangat sedikit."
    graph = penman.decode("(z0 / dan :op1 (z1 / meminta-01 :ARG0 (z2 / aku) :ARG1 (z3 / sial :mod (z4 / hot))) :op2 (z5 / mendapat-01 :ARG0 z2 :ARG1 (z6 / rempah :mod (z7 / sedikit :degree (z8 / sangat)))))")
    source_style = "LABEL_0"
    style_words = ["sedikit"]
    result = sr(text, graph, source_style, style_words)
    assert "_" not in penman.encode(result)
