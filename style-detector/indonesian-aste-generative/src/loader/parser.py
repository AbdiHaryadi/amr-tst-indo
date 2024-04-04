def parse(file, separator):
    sents, labels = [], []
    words = []
    for line in file:
        line = line.strip()
        if line != "":
            words, targets = line.split(separator)
            sents.append(words.split())
            labels.append(eval(targets))
    return sents, labels
