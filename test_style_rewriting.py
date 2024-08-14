from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from style_rewriting import StyleRewriting, TextBasedStyleRewriting
import penman

def test_handle_target_word_with_underscore():
    class DummyStyleRewriting(StyleRewriting):
        def __init__(self):
            super().__init__("dummy_clf_pipeline", "dummy_fasttext_model_path")

        def _load_fasttext_model(self, fasttext_model_path):
            pass

        def _load_clf_pipeline(self, clf_pipeline):
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

def test_text_based_style_rewriting():
    # Di tes ini, diberikan text string, kembalikan string juga, dan pastikan hasilnya beda, dan melibatkan kata yang dimaksud. Pastikan ada "besar hati" di outputnya.
    class DummyTextBasedStyleRewriting(TextBasedStyleRewriting):
        def __init__(self):
            self.max_score_strategy = False
            self.clf_pipe = lambda x: [{"label": "LABEL_1"}]

        def _get_antonym_list(self, word: str):
            if word == "sedikit":
                return ["besar_hati"]
            else:
                raise NotImplementedError
    
    sr = DummyTextBasedStyleRewriting()
    text = "saya meminta thai hot dan mendapatkan rempah yang sangat sedikit."
    source_style = "LABEL_0"
    style_words = ["sedikit"]
    result = sr(text, source_style, style_words)
    assert result == "saya meminta thai hot dan mendapatkan rempah yang sangat besar hati."

def test_style_rewriting_when_clf_error():
    class DummyStyleRewriting(StyleRewriting):
        def __init__(self):
            super().__init__("dummy_clf_pipeline", "dummy_fasttext_model_path")

        def _load_fasttext_model(self, fasttext_model_path):
            pass

        def _load_clf_pipeline(self, clf_pipeline):
            def trigger_error():
                raise ValueError("This error is expected.")
            
            self.clf_pipe = lambda x: trigger_error()

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
    sr(text, graph, source_style, style_words) # Make sure no error, it should be keep running

def test_text_based_style_rewriting_when_clf_error():
    # Di tes ini, diberikan text string, kembalikan string juga, dan pastikan hasilnya beda, dan melibatkan kata yang dimaksud. Pastikan ada "besar hati" di outputnya.
    class DummyTextBasedStyleRewriting(TextBasedStyleRewriting):
        def __init__(self):
            self.max_score_strategy = False

            def trigger_error():
                raise ValueError("This error is expected.")

            self.clf_pipe = lambda x: trigger_error()

        def _get_antonym_list(self, word: str):
            if word == "sedikit":
                return ["besar_hati"]
            else:
                raise NotImplementedError
    
    sr = DummyTextBasedStyleRewriting()
    text = "saya meminta thai hot dan mendapatkan rempah yang sangat sedikit."
    source_style = "LABEL_0"
    style_words = ["sedikit"]
    result = sr(text, source_style, style_words) # Make sure no error, it should be keep running
    assert result == "saya meminta thai hot dan mendapatkan rempah yang sangat besar hati."

def test_text_based_style_rewriting_get_last_log():
    def dummy_clf_pipe_fn(x):
        if "katapertama" in x:
            return [{"label": "LABEL_0", "other_value": 123}]
        elif "katakedua" in x:
            return [{"label": "LABEL_0", "other_value": 456}]
        elif "kataketiga" in x:
            return [{"label": "LABEL_1", "other_value": 789}]
        elif "katakeempat" in x:
            return [{"label": "LABEL_0", "other_value": 1011}]
        else:
            raise NotImplementedError(f"Unexpected word \"{x}\" in dummy_clf_pipe_fn.")
        
    class DummyFastTextModel:
        def __init__(self):
            pass

        def get_nearest_neighbors(self, word, k):
            assert k == 5
            if word == "buruk":
                return [(None, "buruk2")]
            else:
                return []

    class DummyTextBasedStyleRewriting(TextBasedStyleRewriting):
        def __init__(self):
            self.word_expand_size = 5
            self.fasttext_model = DummyFastTextModel()
            self.max_score_strategy = False
            self.ignore_and_warn_if_target_word_not_found = True
            self.clf_pipe = lambda x: dummy_clf_pipe_fn(x)

        def _get_antonym_list(self, word: str):
            if word == "buruk":
                return ["katapertama", "katakedua"]
            elif word == "bagus":
                return ["kataketiga"]
            elif word == "buruk2":
                return ["katakeempat"]
            else:
                raise NotImplementedError(f"Unexpected word \"{word}\" in _get_antonym_list")
    
    sr = DummyTextBasedStyleRewriting()
    text = "itu tidak buruk tetapi tidak bagus juga."
    source_style = "LABEL_0"
    style_words = ["buruk", "bagus"]
    sr(text, source_style, style_words)
    log = sr.get_last_log()
    assert log[0] == {
        "type": "get_antonyms",
        "word": "buruk",
        "antonyms": ["katapertama", "katakedua"]
    }
    assert log[1] == {
        "type": "check_style",
        "text": "itu tidak tetapi tidak juga. katapertama",
        "label": "LABEL_0",
        "other_value": 123
    }
    assert log[2] == {
        "type": "check_style",
        "text": "itu tidak tetapi tidak juga. katakedua",
        "label": "LABEL_0",
        "other_value": 456
    }
    assert log[3] == {
        "type": "expand",
        "old_words": ["buruk"],
        "new_words": ["buruk2"]
    }
    assert log[4] == {
        "type": "get_antonyms",
        "word": "buruk2",
        "antonyms": ["katakeempat"]
    }
    assert log[5] == {
        "type": "check_style",
        "text": "itu tidak tetapi tidak juga. katakeempat",
        "label": "LABEL_0",
        "other_value": 1011
    }
    assert log[6] == {
        "type": "expand",
        "old_words": ["buruk2"],
        "new_words": []
    }
    assert log[7] == {
        "type": "target_word_not_found",
        "text": text,
        "source_style": source_style,
        "word": "buruk"
    }
    assert log[8] == {
        "type": "get_antonyms",
        "word": "bagus",
        "antonyms": ["kataketiga"]
    }
    assert log[9] == {
        "type": "check_style",
        "text": "itu tidak tetapi tidak juga. kataketiga",
        "label": "LABEL_1",
        "other_value": 789
    }
    assert len(log) == 10

def test_style_rewriting_get_last_log():
    def dummy_clf_pipe_fn(x):
        if "katapertama" in x:
            return [{"label": "LABEL_0", "other_value": 123}]
        elif "katakedua" in x:
            return [{"label": "LABEL_0", "other_value": 456}]
        elif "kataketiga" in x:
            return [{"label": "LABEL_1", "other_value": 789}]
        elif "katakeempat" in x:
            return [{"label": "LABEL_0", "other_value": 1011}]
        else:
            raise NotImplementedError(f"Unexpected word \"{x}\" in dummy_clf_pipe_fn.")
        
    class DummyFastTextModel:
        def __init__(self):
            pass

        def get_nearest_neighbors(self, word, k):
            if word == "buruk":
                return [(None, "buruk2")]
            else:
                return []
    
    class DummyStyleRewriting(StyleRewriting):
        def __init__(self):
            super().__init__("dummy_clf_pipeline", "dummy_fasttext_model_path", remove_polarity_strategy=False)

        def _load_fasttext_model(self, fasttext_model_path):
            self.fasttext_model = DummyFastTextModel()

        def _load_clf_pipeline(self, clf_pipeline):
            self.clf_pipe = lambda x: dummy_clf_pipe_fn(x)

        def _get_antonym_list(self, word: str):
            if word == "buruk":
                return ["katapertama", "katakedua"]
            elif word == "bagus":
                return ["kataketiga"]
            elif word == "buruk2":
                return ["katakeempat"]
            else:
                raise NotImplementedError(f"Unexpected word \"{word}\" in _get_antonym_list")
    
    sr = DummyStyleRewriting()
    text = "itu tidak buruk tetapi tidak bagus juga."
    amr_str = """
        ( k / kontras-01
            :ARG1 ( b / buruk-00
                :polarity -
                :ARG1 ( i / itu ) )
            :ARG2 ( b2 / bagus-00
                :polarity -
                :ARG1 i
                :mod ( j / juga ) ) )
    """
    amr = penman.decode(amr_str)
    source_style = "LABEL_0"
    style_words = ["buruk", "bagus"]
    sr(text, amr, source_style, style_words)
    log = sr.get_last_log()
    assert log[0] == {
        "type": "get_antonyms",
        "word": "buruk",
        "antonyms": ["katapertama", "katakedua"]
    }
    assert log[1] == {
        "type": "check_style",
        "text": "itu tidak tetapi tidak juga. katapertama",
        "label": "LABEL_0",
        "other_value": 123
    }
    assert log[2] == {
        "type": "check_style",
        "text": "itu tidak tetapi tidak juga. katakedua",
        "label": "LABEL_0",
        "other_value": 456
    }
    assert log[3] == {
        "type": "expand",
        "old_words": ["buruk"],
        "new_words": ["buruk2"]
    }
    assert log[4] == {
        "type": "get_antonyms",
        "word": "buruk2",
        "antonyms": ["katakeempat"]
    }
    assert log[5] == {
        "type": "check_style",
        "text": "itu tidak tetapi tidak juga. katakeempat",
        "label": "LABEL_0",
        "other_value": 1011
    }
    assert log[6] == {
        "type": "expand",
        "old_words": ["buruk2"],
        "new_words": []
    }
    assert log[7] == {
        "type": "target_word_not_found",
        "text": text,
        "source_style": source_style,
        "word": "buruk"
    }
    assert log[8] == {
        "type": "get_antonyms",
        "word": "bagus",
        "antonyms": ["kataketiga"]
    }
    assert log[9] == {
        "type": "check_style",
        "text": "itu tidak tetapi tidak juga. kataketiga",
        "label": "LABEL_1",
        "other_value": 789
    }
    assert log[10] == {
        "type": "rewrite_node",
        "old_amr": penman.encode(amr, indent=None),
        "new_amr": penman.encode(penman.decode(amr_str.replace("bagus-00", "kataketiga-00")), indent=None)
    }
    assert len(log) == 11

def test_style_rewriting_use_sastrawi_for_node_match():
    stemmer = StemmerFactory().create_stemmer()
    assert stemmer.stem("mendatangi") == stemmer.stem("datang")

    class DummyStyleRewriting(StyleRewriting):
        def __init__(self):
            super().__init__("dummy_clf_pipeline", "dummy_fasttext_model_path")

        def _load_fasttext_model(self, fasttext_model_path):
            pass

        def _load_clf_pipeline(self, clf_pipeline):
            self.clf_pipe = lambda x: [{"label": "LABEL_1"}]

        def _get_antonym_list(self, word: str):
            raise NotImplementedError("You should only use polarity check, right?")

    sr = DummyStyleRewriting()
    text = "saya tidak akan datang ke salon ini lagi."
    graph = penman.decode("(z0 / mendatangi-01 :polarity - :ARG1 (z1 / aku) :ARG4 (z2 / salon :mod (z3 / ini)) :mod (z4 / lagi))")
    source_style = "LABEL_0"
    style_words = ["datang"]
    result = sr(text, graph, source_style, style_words)
    assert penman.encode(result, indent=None) == penman.encode(
        penman.decode("(z0 / mendatangi-01 :ARG1 (z1 / aku) :ARG4 (z2 / salon :mod (z3 / ini)) :mod (z4 / lagi))"),
        indent=None
    )

def test_style_rewriting_not_use_sastrawi_for_node_match():
    class DummyStyleRewriting(StyleRewriting):
        def __init__(self):
            super().__init__("dummy_clf_pipeline", "dummy_fasttext_model_path", use_stem=False)

        def _load_fasttext_model(self, fasttext_model_path):
            pass

        def _load_clf_pipeline(self, clf_pipeline):
            self.clf_pipe = lambda x: [{"label": "LABEL_1"}]

        def _get_antonym_list(self, word: str):
            raise NotImplementedError("No match, right?")

    sr = DummyStyleRewriting()
    text = "saya tidak akan datang ke salon ini lagi."
    graph = penman.decode("(z0 / mendatangi-01 :polarity - :ARG1 (z1 / aku) :ARG4 (z2 / salon :mod (z3 / ini)) :mod (z4 / lagi))")
    source_style = "LABEL_0"
    style_words = ["datang"]
    result = sr(text, graph, source_style, style_words)
    assert penman.encode(result, indent=None) == penman.encode(graph, indent=None)

def test_style_rewriting_replace_instance():
    stemmer = StemmerFactory().create_stemmer()
    assert stemmer.stem("memburuk") == stemmer.stem("buruk")

    class DummyStyleRewriting(StyleRewriting):
        def __init__(self):
            super().__init__("dummy_clf_pipeline", "dummy_fasttext_model_path")

        def _load_fasttext_model(self, fasttext_model_path):
            pass

        def _load_clf_pipeline(self, clf_pipeline):
            self.clf_pipe = lambda x: [{"label": "LABEL_1"}]

        def _get_antonym_list(self, word: str):
            if word == "memburuk":
                return ["bersih"]
            else:
                raise NotImplementedError(f"Unexpected word \"{word}\" in _get_antonym_list")

    sr = DummyStyleRewriting()
    text = "semenjak joes berganti kepemilikan, kondisinya semakin memburuk."
    graph = penman.decode("""(z0 / memiliki-derajat-91 :ARG1 (z1 / kondisi :poss (z2 / dia)) :ARG2 (z3 / buruk-07 :ARG1-of (z4 / meningkat-02)) :ARG3 (z5 / lebih) :time (z6 / setelah :op1 (z7 / pindah-01 :ARG0 (z8 / orang :wiki - :name (z9 / nama :op1 "joes")) :ARG1 (z10 / sendiri-01 :ARG0 z8))))""")
    source_style = "LABEL_0"
    style_words = ["memburuk"]
    result = sr(text, graph, source_style, style_words)
    assert penman.encode(result, indent=None) == penman.encode(
        penman.decode("""(z0 / memiliki-derajat-91 :ARG1 (z1 / kondisi :poss (z2 / dia)) :ARG2 (z3 / bersih-00 :ARG1-of (z4 / meningkat-02)) :ARG3 (z5 / lebih) :time (z6 / setelah :op1 (z7 / pindah-01 :ARG0 (z8 / orang :wiki - :name (z9 / nama :op1 "joes")) :ARG1 (z10 / sendiri-01 :ARG0 z8))))"""),
        indent=None
    )

def test_style_rewriting_replace_instance_without_reset_sense():
    stemmer = StemmerFactory().create_stemmer()
    assert stemmer.stem("memburuk") == stemmer.stem("buruk")

    class DummyStyleRewriting(StyleRewriting):
        def __init__(self):
            super().__init__("dummy_clf_pipeline", "dummy_fasttext_model_path", reset_sense_strategy=False)

        def _load_fasttext_model(self, fasttext_model_path):
            pass

        def _load_clf_pipeline(self, clf_pipeline):
            self.clf_pipe = lambda x: [{"label": "LABEL_1"}]

        def _get_antonym_list(self, word: str):
            if word == "memburuk":
                return ["bersih"]
            else:
                raise NotImplementedError(f"Unexpected word \"{word}\" in _get_antonym_list")

    sr = DummyStyleRewriting()
    text = "semenjak joes berganti kepemilikan, kondisinya semakin memburuk."
    graph = penman.decode("""(z0 / memiliki-derajat-91 :ARG1 (z1 / kondisi :poss (z2 / dia)) :ARG2 (z3 / buruk-07 :ARG1-of (z4 / meningkat-02)) :ARG3 (z5 / lebih) :time (z6 / setelah :op1 (z7 / pindah-01 :ARG0 (z8 / orang :wiki - :name (z9 / nama :op1 "joes")) :ARG1 (z10 / sendiri-01 :ARG0 z8))))""")
    source_style = "LABEL_0"
    style_words = ["memburuk"]
    result = sr(text, graph, source_style, style_words)
    assert penman.encode(result, indent=None) == penman.encode(
        penman.decode("""(z0 / memiliki-derajat-91 :ARG1 (z1 / kondisi :poss (z2 / dia)) :ARG2 (z3 / bersih-07 :ARG1-of (z4 / meningkat-02)) :ARG3 (z5 / lebih) :time (z6 / setelah :op1 (z7 / pindah-01 :ARG0 (z8 / orang :wiki - :name (z9 / nama :op1 "joes")) :ARG1 (z10 / sendiri-01 :ARG0 z8))))"""),
        indent=None
    )

def test_style_rewriting_replace_non_frame_instance():
    stemmer = StemmerFactory().create_stemmer()
    assert stemmer.stem("memburuk") == stemmer.stem("buruk")

    class DummyStyleRewriting(StyleRewriting):
        def __init__(self):
            super().__init__("dummy_clf_pipeline", "dummy_fasttext_model_path")

        def _load_fasttext_model(self, fasttext_model_path):
            pass

        def _load_clf_pipeline(self, clf_pipeline):
            self.clf_pipe = lambda x: [{"label": "LABEL_1"}]

        def _get_antonym_list(self, word: str):
            if word == "memburuk":
                return ["bersih"]
            else:
                raise NotImplementedError(f"Unexpected word \"{word}\" in _get_antonym_list")

    sr = DummyStyleRewriting()
    text = "semenjak joes berganti kepemilikan, kondisinya semakin memburuk."
    graph = penman.decode("""(z0 / memiliki-derajat-91 :ARG1 (z1 / kondisi :poss (z2 / dia)) :ARG2 (z3 / buruk :ARG1-of (z4 / meningkat-02)) :ARG3 (z5 / lebih) :time (z6 / setelah :op1 (z7 / pindah-01 :ARG0 (z8 / orang :wiki - :name (z9 / nama :op1 "joes")) :ARG1 (z10 / sendiri-01 :ARG0 z8))))""")
    source_style = "LABEL_0"
    style_words = ["memburuk"]
    result = sr(text, graph, source_style, style_words)
    assert penman.encode(result, indent=None) == penman.encode(
        penman.decode("""(z0 / memiliki-derajat-91 :ARG1 (z1 / kondisi :poss (z2 / dia)) :ARG2 (z3 / bersih :ARG1-of (z4 / meningkat-02)) :ARG3 (z5 / lebih) :time (z6 / setelah :op1 (z7 / pindah-01 :ARG0 (z8 / orang :wiki - :name (z9 / nama :op1 "joes")) :ARG1 (z10 / sendiri-01 :ARG0 z8))))"""),
        indent=None
    )
