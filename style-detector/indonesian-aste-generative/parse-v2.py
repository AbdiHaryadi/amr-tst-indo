import os


ANNOTATION_SEPERATOR = "####"
LABEL_SEPERATOR = ", "

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


def remove_implicit(labels):
    explicit_labels = []
    for triplets in labels:
        res = []
        for triplet in triplets:
            if triplet[0][0] != -1:
                res.append(triplet)
        explicit_labels.append(res)
    return explicit_labels

def write_parse_result(sents, labels, target_path):
    """
    Helper func to output the correct annotated format for OTE-MTL framework
    """
    with open(target_path, "w") as fout:
        for text, labels in zip(sents, labels):
            fout.write(text + ANNOTATION_SEPERATOR)
            print(labels)
            fout.write(str(labels) + "\n")



if __name__ == "__main__":
    PROCESSED_DATA_UNFILTER_DIR = "dataset/v3"
    TARGET_DATA_UNFILTER_DIR = "dataset/v2"

    train_path = os.path.join(PROCESSED_DATA_UNFILTER_DIR, "train.txt" )
    test_path = os.path.join(PROCESSED_DATA_UNFILTER_DIR, "test.txt")
    val_path = os.path.join(PROCESSED_DATA_UNFILTER_DIR, "dev.txt")

    target_train_path = os.path.join(TARGET_DATA_UNFILTER_DIR, "train.txt" )
    target_test_path = os.path.join(TARGET_DATA_UNFILTER_DIR, "test.txt")
    target_val_path = os.path.join(TARGET_DATA_UNFILTER_DIR, "dev.txt")

    with open(train_path, 'r') as f:
        sents, labels = parse(f, ANNOTATION_SEPERATOR )
        labels = remove_implicit(labels)
        write_parse_result(sents, labels, target_train_path)

    
    with open(val_path, 'r') as f:
        sents, labels = parse(f, ANNOTATION_SEPERATOR )
        labels = remove_implicit(labels)
        write_parse_result(sents, labels, target_val_path)

    
    with open(test_path, 'r') as f:
        sents, labels = parse(f, ANNOTATION_SEPERATOR )
        labels = remove_implicit(labels)
        write_parse_result(sents, labels, target_test_path)



