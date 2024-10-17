import evaluate
bleu_metric = evaluate.load("evaluate-metric/sacrebleu")
rouge_metric = evaluate.load("rouge")

def compute_metrics_generation(decoded_preds, decoded_labels):
    return compute_bleu(decoded_preds, decoded_labels)

def compute_bleu(raw_predictions, raw_references):
    # Some simple post-processing
    predictions = [pred.strip() for pred in raw_predictions]
    references = [[label.strip()] for label in raw_references] # sacrebleu uses multi reference setting

    return bleu_metric.compute(
        predictions=predictions, references=references, lowercase=True
    )

def compute_rouge_l(raw_predictions, raw_references):
    return rouge_metric.compute(
        predictions=raw_predictions, references=raw_references, rouge_types=["rougeL"]
    )

def compute_detail_bleu(raw_predictions, raw_references):
    results = []
    for raw_p, raw_r in zip(raw_predictions, raw_references):
        single_result = compute_bleu([raw_p], [raw_r])
        results.append(single_result)
    
    return results

def compute_detail_rouge_l(raw_predictions, raw_references):
    results = []
    for raw_p, raw_r in zip(raw_predictions, raw_references):
        single_result = compute_rouge_l([raw_p], [raw_r])
        results.append(single_result)
    
    return results
