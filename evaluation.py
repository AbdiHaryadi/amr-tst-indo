import evaluate
bleu_metric = evaluate.load("evaluate-metric/sacrebleu")

def compute_metrics_generation(decoded_preds, decoded_labels):
    # Some simple post-processing
    predictions = [pred.strip() for pred in decoded_preds]
    references = [[label.strip()] for label in decoded_labels] # sacrebleu uses multi reference setting

    return bleu_metric.compute(
        predictions=predictions, references=references, lowercase=True
    )
