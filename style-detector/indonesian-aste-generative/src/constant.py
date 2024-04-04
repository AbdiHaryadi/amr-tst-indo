import os

GENERAL_ASPECT = "_"
GENERAL_ASPECTS = ["_", "hotel", "kamar", "pelayanan", "tempat", "menginap", "lokasi"]

class ModelType:
    T5Model = "t5"


class ProcessType:
    DoTrain = "train"
    DoTest = "test"


class DatasetType:
    FILTERED = "filter"
    UNFILTERED = "unfilter"
    ANNOTATION = "implicit-v2"


class PostprocessType:
    EDITDISTANCE = "edit-distance"
    EMBEDDING = "embedding"

class OptimizerType:
    ADAMW = "adamw"
    ADAFACTOR = "adafactor"


class Path:
    RESOURCES = "resources"
    DATA = "data"
    PRETRAINED = "pretrained"
    MODEL = "bin"
    REPORT = "reports"

    TRAIN = "train.txt"
    TEST = "test.txt"
    VAL = "dev.txt"

    TRAIN_FILTERED_PATH = os.path.join(DATA, "processed", DatasetType.FILTERED, TRAIN)
    TEST_FILTERED_PATH = os.path.join(DATA, "processed", DatasetType.FILTERED, TEST)
    VAL_FILTERED_PATH = os.path.join(DATA, "processed", DatasetType.FILTERED, VAL)

    TRAIN_UNFILTERED_PATH = os.path.join(DATA, "processed", DatasetType.UNFILTERED, TRAIN)
    TEST_UNFILTERED_PATH = os.path.join(DATA, "processed", DatasetType.UNFILTERED, TEST)
    VAL_UNFILTERED_PATH = os.path.join(DATA, "processed", DatasetType.UNFILTERED, VAL)

    TRAIN_ANNOTATION_PATH = os.path.join(DATA, "processed", DatasetType.ANNOTATION, TRAIN)
    TEST_ANNOTATION_PATH = os.path.join(DATA, "processed", DatasetType.ANNOTATION, TEST)
    VAL_ANNOTATION_PATH = os.path.join(DATA, "processed", DatasetType.ANNOTATION, VAL)
