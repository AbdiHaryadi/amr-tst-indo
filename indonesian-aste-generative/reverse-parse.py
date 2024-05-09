import os
import json

BIOTAG_SEPERATOR = "//"
ANNOTATION_SEPERATOR = "####"
TOKEN_SEPERATOR = " "

reverse_polarity_annotation_map = {
    "NEU": "NT",
    "NEG": "NG",
    "POS": "PO",
}


def parse(file, separator):
    sents, labels = [], []
    words = []
    for line in file:
        line = line.strip()
        if line != "":
            words, targets = line.split(separator)
            sents.append(words)
            labels.append(eval(targets))
    return sents, labels


def load_data_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


def extract_term_tag(tags):
    for tag in tags:
        if tag.startswith("ASPECT") or tag.startswith("SENTIMENT"):
            return tag
    return None

def generate_bio_token(tokens, labels):
    return [
        token + BIOTAG_SEPERATOR + term_tag
        for token, term_tag in zip(tokens, get_all_term(tokens, labels))
    ]

def get_all_term(tokens, labels):
    prev = ""
    term_bio = []
    for i, token in enumerate(tokens):
        if i in labels:
            if prev == "":
                term_bio.append("B")
                prev = term_bio[-1]
            else:
                term_bio.append("I")
        else:
            term_bio.append("O")
    return term_bio


def convert_json(sentence, all_aspect_tags, all_sent_tags, valid, tokens, label):

    curr_triplets = []

    def handle_triple(tokens, labels):
        aspect_tags = labels[0]
        sentiment_tags = labels[1]
        polarity = reverse_polarity_annotation_map[labels[2]]
        aspect_tags = " ".join(generate_bio_token(tokens, aspect_tags))
        sent_tags = " ".join(generate_bio_token(tokens, sentiment_tags))
        return {
            "aspect_tags": aspect_tags,
            "sent_tags": sent_tags,
            "polarity": polarity,
        }

    for triplet in label:
        curr_triplets.append(handle_triple(tokens, triplet))

    return {
        "sentence": sentence,
        "aspect_tags": all_aspect_tags,
        "sent_tags": all_sent_tags,
        "triples": curr_triplets,
        "valid": valid,
    }

def load_and_parse(source, labels):
    res = []

    for i, triplets in enumerate(labels):
        sentence = source[i]["sentence"]
        aspect_tags = source[i]["aspect_tags"]
        sent_tags = source[i]["sent_tags"]
        valid = source[i]["valid"]
        res.append(convert_json(sentence, aspect_tags, sent_tags, valid, sentence.split(TOKEN_SEPERATOR), triplets))
    return res

if __name__ == "__main__":
    PROCESSED_DATA_UNFILTER_DIR = "dataset/v2"
    SOURCE_DATA_DIR = "dataset/v0"

    train_path = os.path.join(PROCESSED_DATA_UNFILTER_DIR, "train.txt")
    test_path = os.path.join(PROCESSED_DATA_UNFILTER_DIR, "test.txt")
    val_path = os.path.join(PROCESSED_DATA_UNFILTER_DIR, "dev.txt")

    target_train_path = os.path.join(PROCESSED_DATA_UNFILTER_DIR, "train.json")
    target_test_path = os.path.join(PROCESSED_DATA_UNFILTER_DIR, "test.json")
    target_val_path = os.path.join(PROCESSED_DATA_UNFILTER_DIR, "dev.json")

    source_train_path = os.path.join(SOURCE_DATA_DIR, "train.json")
    source_test_path = os.path.join(SOURCE_DATA_DIR, "test.json")
    source_val_path = os.path.join(SOURCE_DATA_DIR, "validation.json")

    source_train_data = load_data_json(source_train_path)
    source_test_data = load_data_json(source_test_path)
    source_val_data = load_data_json(source_val_path)

    with open(train_path, "r") as f:
        sents, labels = parse(f, ANNOTATION_SEPERATOR)
        res = load_and_parse(source_train_data, labels)
    
    with open(target_train_path, "w") as outfile:
        json.dump(res, outfile)

    # print("Training Data")
    # for i in range(len(res)):
    #     if res[i] != source_train_data[i]:
    #         print(i)
    # print("---")

    with open(val_path, "r") as f:
        sents, labels = parse(f, ANNOTATION_SEPERATOR)
        res = load_and_parse(source_val_data, labels)
    
    with open(target_val_path, "w") as outfile:
        json.dump(res, outfile)

    # print("Validation Data")
    # for i in range(len(res)):
    #     if res[i] != source_val_data[i]:
    #         print(i)
    # print("---")

    with open(test_path, "r") as f:
        sents, labels = parse(f, ANNOTATION_SEPERATOR)
        res = load_and_parse(source_test_data, labels)
    
    with open(target_test_path, "w") as outfile:
        json.dump(res, outfile)

    # print("Test Data")
    # for i in range(len(res)):
    #     if res[i] != source_test_data[i]:
    #         print(i)
    # print("---")

    

    
