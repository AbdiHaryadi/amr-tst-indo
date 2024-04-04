import os
import csv
import json

TSV_SEPERATOR = "\t"

BIOTAG_SEPERATOR = "//"
TOKEN_SEPERATOR = " "
ANNOTATION_SEPERATOR = "####"
LABEL_SEPERATOR = ", "

polarity_annotation_map = {
    "NT": "NEU",
    "NG": "NEG",
    "PO": "POS",
}

def get_elmt_idx(data):
    """
    Helper func to help identify the start & end idx from IOB Tagging ///O
    """
    splitter = lambda val : val[1:].split(BIOTAG_SEPERATOR)
    tag = [splitter(datum) for datum in data]
    for i in range(len(tag)):
        tag[i][0] = data[i][0] + tag[i][0]
    start_idx, end_idx, found = -1, -1, False
    for idx, word_tag in enumerate(tag):
        _, tag = word_tag
        if tag == "B":
            start_idx = idx
            end_idx = idx
            found = True
        elif tag == "I" and found:
            end_idx = idx
        elif tag == "O" and found:
            end_idx = idx - 1
            break
    return start_idx, end_idx

def load_data_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def check_sentence_pack(term_dict):
    return all(term_dict.values())

def extract_term_tag(tags):
    for tag in tags:
        if tag.startswith("ASPECT") or tag.startswith("SENTIMENT"):
            return tag
    return None

def handle_relation(relation):
    start_bracket = relation.index("[")
    end_bracket = relation.index("]")
    polarity = relation[:start_bracket]
    pair = relation[start_bracket + 1 : end_bracket]
    sent, aspect = pair.split("_")
    return polarity, sent, aspect

def get_all_term(tags, term):
    prev = ""
    term_bio = []
    for tag in tags:
        if term in tag:
            if tag != prev:
                term_bio.append("B")
                prev = tag
            else:
                term_bio.append("I")
        else:
            term_bio.append("O")
    return term_bio

def generate_bio_token(tokens, term_tags, term):
    return [
        token + BIOTAG_SEPERATOR + term_tag
        for token, term_tag in zip(tokens, get_all_term(term_tags, term))
    ]

def load_raw_data(filename):
    data, labels = [], []
    with open(filename) as file:
        read_tsv = csv.reader(file, delimiter=TSV_SEPERATOR, quoting=csv.QUOTE_NONE)
        tokens, tags = [], []
        for row in read_tsv:
            if len(row) > 1:
                tokens.append(row[2])
                tags.append(row[3])
            else:
                if len(tokens) > 0:
                    data.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
    return data, labels

def convert_json(tokens, tags):
    sentence = " ".join(tokens)
    triple = []
    term_tags = []
    relation_tags = []
    for tag in tags:
        tag = tag.split("|")
        term_tag = tag[0] if len(tag) == 1 else extract_term_tag(tag)
        # If term tag exist
        if term_tag != None:
            term_tags.append(term_tag)
            tag.pop(tag.index(term_tag))
            relation_tags += tag

    # All Terms
    all_aspect_tags = TOKEN_SEPERATOR.join(
        generate_bio_token(tokens, term_tags, "ASPECT")
    )
    all_sent_tags = TOKEN_SEPERATOR.join(
        generate_bio_token(tokens, term_tags, "SENTIMENT")
    )

    # Handle Triple
    term_dict = {term_tag: False for term_tag in term_tags if term_tag != "_"}

    def handle_triple(tokens, term_tags, relation_tag):
        polarity, term_num_1, term_num_2 = handle_relation(relation_tag)
        # Handle switched term num (Each term have unique number)
        if f"SENTIMENT[{term_num_2}]" in term_dict:
            sent_num, aspect_num = term_num_2, term_num_1
        else:
            sent_num, aspect_num = term_num_1, term_num_2

        sent_term = f"SENTIMENT[{sent_num}]"
        aspect_term = f"ASPECT[{aspect_num}]"
        term_dict[aspect_term] = True
        term_dict[sent_term] = True
        aspect_tags = " ".join(generate_bio_token(tokens, term_tags, aspect_term))
        sent_tags = " ".join(generate_bio_token(tokens, term_tags, sent_term))
        return {
            "aspect_tags": aspect_tags,
            "sent_tags": sent_tags,
            "polarity": polarity,
        }

    for relation_tag in relation_tags:
        triple.append(handle_triple(tokens, term_tags, relation_tag))

    valid = check_sentence_pack(term_dict) and len(triple) != 0

    return {
        "sentence": sentence,
        "aspect_tags": all_aspect_tags,
        "sent_tags": all_sent_tags,
        "triples": triple,
        "valid": valid,
    }

def parse_raw_batch(input_path, output_path, check_validity=False):
    """
    Wrapper to parse raw data format into interrim data (json formatted) triplet annotated data
    """
    files = os.listdir(input_path)
    json_datas = []
    filenames = []
    for file in files:
        in_filename = os.path.join(input_path, file)
        out_filename = os.path.join(output_path, file.replace(".tsv", ".json"))
        data, labels = load_raw_data(in_filename)
        json_data = []
        for tokens, tags in zip(data, labels):
            data = convert_json(tokens, tags)
            if check_validity and data["valid"] == False:
                continue
            json_data.append(data)
        json_datas.append(json_data)
        filenames.append(out_filename)
    return json_datas, filenames

def write_file_batch(json_datas, filenames):
    """
    Wrapper to save batch interrim data (json formatted) triplet annotated data
    """
    for filename, json_data in zip(filenames, json_datas):
        write_file(json_data, filename)

def write_file(json_data, filename):
    with open(filename, "w") as outfile:
        json.dump(json_data, outfile)

def parse_interim_batch(input_path, output_path):
    """
    Wrapper to parse batched data on each file
    """
    files = os.listdir(input_path)
    filenames = []
    parsed_datas = []
    for file in files:
        in_filename = os.path.join(input_path, file)
        out_filename = os.path.join(output_path, file.replace(".json", ".txt"))

        parsed_datas.append(parse_interim(load_data_json(in_filename)))
        filenames.append(out_filename)
    return parsed_datas, filenames

def parse_interim(data, valid_only=False):
    """
    Wrapper to parse interrim data (json formatted) into correct annotated data for OTE-MTL framework
    """
    parsed_data = []
    for datum in data:
        triplets = []
        if valid_only and not datum.get("valid"):
            pass
        else:
            for triplet in datum.get("triples"):
                aspect_start_idx, aspect_end_idx = get_elmt_idx(
                    triplet.get("aspect_tags").split(TOKEN_SEPERATOR)
                )

                sentiment_start_idx, sentiment_end_idx = get_elmt_idx(
                    triplet.get("sent_tags").split(TOKEN_SEPERATOR)
                )

                polarity = triplet.get("polarity")
                triplets.append(
                    str(
                        (
                            get_iterate_idx(aspect_start_idx, aspect_end_idx),
                            get_iterate_idx(sentiment_start_idx, sentiment_end_idx),
                            polarity_annotation_map.get(polarity),
                        )
                    )
                )
        sentence = datum.get("sentence")
        parsed_data.append([sentence, triplets])
    return parsed_data

def get_iterate_idx(start_idx, end_idx):
    assert start_idx <= end_idx
    return [i for i in range(start_idx, end_idx + 1)]

def write_parse_result(parsed_data, target_path):
    """
    Helper func to output the correct annotated format for OTE-MTL framework
    """
    with open(target_path, "w") as fout:
        for parsed_datum in parsed_data:
            text, labels = parsed_datum

            fout.write(text + ANNOTATION_SEPERATOR)
            fout.write("[" + LABEL_SEPERATOR.join(labels) + "]" + "\n")

def write_parsed_batch(parsed_datas, filenames):
    """
    Wrapper to save batch processed data (Sem v2) triplet annotated data
    """
    for filename, parsed_data in zip(filenames, parsed_datas):
        write_parse_result(parsed_data, filename)

if __name__ == "__main__":
    RAW_DATA_DIR = "data/raw"
    INTERIM_DATA_FILTER_DIR = "data/interim/filter"
    INTERIM_DATA_UNFILTER_DIR = "data/interim/unfilter"
    PROCESSED_DATA_FILTER_DIR = "dataset/processed/filter"
    PROCESSED_DATA_UNFILTER_DIR = "dataset/v1"

    # == Save Interim data ==
    json_datas, filenames = parse_raw_batch(os.path.join(RAW_DATA_DIR), os.path.join(INTERIM_DATA_UNFILTER_DIR), check_validity=False)
    write_file_batch(json_datas=json_datas, filenames=filenames)

    # json_datas, filenames = parse_raw_batch(os.path.join(RAW_DATA_DIR), os.path.join(INTERIM_DATA_FILTER_DIR), check_validity=True)
    # write_file_batch(json_datas=json_datas, filenames=filenames)

    parsed_datas, filenames = parse_interim_batch(os.path.join(INTERIM_DATA_UNFILTER_DIR), os.path.join(PROCESSED_DATA_UNFILTER_DIR))
    write_parsed_batch(parsed_datas=parsed_datas, filenames=filenames)

    # parsed_datas, filenames = parse_interim_batch(os.path.join(INTERIM_DATA_FILTER_DIR), os.path.join(PROCESSED_DATA_FILTER_DIR))
    # write_parsed_batch(parsed_datas=parsed_datas, filenames=filenames)


