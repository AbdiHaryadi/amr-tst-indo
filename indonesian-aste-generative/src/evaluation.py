import os

import pandas as pd
from tqdm import tqdm
from src.postprocess import IPostprocess

from src.utility import extract, get_device
from src.constant import GENERAL_ASPECT, GENERAL_ASPECTS, DatasetType, Path

# == Metrics ==
precision_fn = lambda n_tp, n_pred: float(n_tp) / float(n_pred) if n_pred != 0 else 0
recall_fn = lambda n_tp, n_gold: float(n_tp) / float(n_gold) if n_gold != 0 else 0
f1_fn = (
    lambda precision, recall: (2 * precision * recall) / (precision + recall)
    if precision != 0 or recall != 0
    else 0
)


def score(pred, gold):
    assert len(pred) == len(gold)
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred)):
        n_gold += len(gold[i])
        n_pred += len(pred[i])

        for t in pred[i]:
            if t in gold[i]:
                n_tp += 1
            

    precision = precision_fn(n_tp, n_pred)
    recall = recall_fn(n_tp, n_gold)
    f1 = f1_fn(precision, recall)

    return {"precision": precision, "recall": recall, "f1": f1}

## Unused...
def check_implicit_aspect(t1, coll_t2):
    for t2 in coll_t2:
        if t1[1] == t2[1] and t1[2] == t2[2] and t2[0] == GENERAL_ASPECT and t1[0] in GENERAL_ASPECTS:
            return True 
    return False 

## Unused....
def score_include_implicit(pred, gold):
    assert len(pred) == len(gold)
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred)):
        n_gold += len(gold[i])
        n_pred += len(pred[i])

        for t in pred[i]:
            if t in gold[i] or check_implicit_aspect(t, gold[i]):
                n_tp += 1

    precision = precision_fn(n_tp, n_pred)
    recall = recall_fn(n_tp, n_gold)
    f1 = f1_fn(precision, recall)

    return {"precision": precision, "recall": recall, "f1": f1}


# == Evaluation ==
def evaluate(pred_seqs, gold_seqs, postprocessor: IPostprocess, sents, implicit=False):
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract(gold_seqs[i])
        pred_list = extract(pred_seqs[i])

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    raw_scores = score(all_preds, all_labels)

    all_fixed_preds = postprocessor.check_and_fix_preds(all_preds, sents, implicit)
    fixed_scores = score(all_fixed_preds, all_labels)

    return raw_scores, fixed_scores, all_labels, all_preds, all_fixed_preds


class Evaluator:
    def __init__(self, postprocessor, configs):
        self.configs = configs
        self.postprocessor: IPostprocess = postprocessor
        self.device = get_device()

    def evaluate(self, tokenizer, model, loader, sents):
        model.eval()
        mode = self.configs.get("loader").get("mode")
        outputs, targets = [], []
        max_sequence_length = self.configs.get("loader").get("max_seq_length")
        with tqdm(loader, unit="batch") as tevaluator:
            for batch in tevaluator:
                tevaluator.set_description("Generating")
                outs = model.generate(
                    input_ids=batch["source_ids"].to(self.device),
                    attention_mask=batch["source_mask"].to(self.device),
                    max_length=max_sequence_length,
                )

                dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
                target = [
                    tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in batch["target_ids"]
                ]

                outputs.extend(dec)
                targets.extend(target)

            raw_scores, fixed_scores, all_labels, all_preds, all_fixed_preds = evaluate(
                outputs, targets, self.postprocessor, sents, implicit=mode==DatasetType.ANNOTATION
            )

        return {
            "raw_scores": raw_scores,
            "fixed_scores": fixed_scores,
            "all_labels": all_labels,
            "all_preds": all_preds,
            "all_fixed_preds": all_fixed_preds,
        }

    def export(self, type, prefix, tokenizer, model, loader, sents):
        path = os.path.join(Path.MODEL, prefix, type)
        os.makedirs(path, exist_ok=True)

        df, raw_score, fixed_score = self.evaluation_summary_result(
            tokenizer, model, loader, sents
        )

        exploded_cols = [
            "idx",
            "text",
            "all_labels",
            "all_preds",
            "all_fixed_preds",
            "matching_comments",
        ]
        exploded_df = df[exploded_cols].explode(
            ["all_labels", "all_preds", "all_fixed_preds", "matching_comments"]
        )

        triplet_false_mask = exploded_df["all_fixed_preds"] != exploded_df["all_labels"]

        false_exploded_df = exploded_df[triplet_false_mask]

        df.to_csv(os.path.join(path, "result.csv"), index=False)
        exploded_df.to_csv(os.path.join(path, "triplets.csv"), index=False)
        false_exploded_df.to_csv(os.path.join(path, "false-triplets.csv"), index=False)

        metric_summary_df = pd.DataFrame(
            {
                "metrics": ["precision", "recall", "f1"],
                "raw_score": [
                    raw_score["precision"],
                    raw_score["recall"],
                    raw_score["f1"],
                ],
                "fixed_score": [
                    fixed_score["precision"],
                    fixed_score["recall"],
                    fixed_score["f1"],
                ],
            }
        )
        metric_summary_df.to_csv(os.path.join(path, "score.csv"), index=False)

    def evaluation_summary_result(self, model, tokenizer, loader, sents):
        eval_res = self.evaluate(model, tokenizer, loader, sents)
        assert (
            len(sents)
            == len(eval_res["all_labels"])
            == len(eval_res["all_preds"])
            == len(eval_res["all_fixed_preds"])
        )

        true, false = [0] * len(sents), [0] * len(sents)
        for idx, label in enumerate(eval_res["all_labels"]):
            for t in eval_res["all_preds"][idx]:
                if t in label:
                    true[idx] += 1
                else:
                    false[idx] += 1

        all_labels = []
        all_preds = []
        all_fixed_preds = []
        all_matching_comments = []
        for idx in tqdm(range(len(sents))):
            is_diff = false[idx] > 0
            (
                new_label,
                new_pred,
                new_fixed,
                matching_comments,
            ) = self.__matching_solution(
                eval_res["all_labels"][idx],
                eval_res["all_preds"][idx],
                eval_res["all_fixed_preds"][idx],
            )
            all_labels.append(new_label)
            all_preds.append(new_pred)
            all_fixed_preds.append(new_fixed)
            all_matching_comments.append(matching_comments)

        res_df = pd.DataFrame(
            {
                "idx": [_ for _ in range(len(sents))],
                "text": [" ".join(elem) for elem in sents],
                "all_labels": all_labels,
                "all_preds": all_preds,
                "all_fixed_preds": all_fixed_preds,
                "matching_comments": all_matching_comments,
                "num_true": true,
                "num_false": false,
            }
        )

        raw_metric_score = eval_res["raw_scores"]
        fixed_metric_score = eval_res["fixed_scores"]

        return res_df, raw_metric_score, fixed_metric_score

    def __matching_solution(self, label, pred, fixed):
        n = max(len(label), len(pred))
        new_label = ["-"] * n
        new_pred = ["-"] * n
        new_fixed = ["-"] * n
        matching_comments = ["-"] * n

        # == Sorting ==
        sorting_fn = lambda triplet: (triplet[0], triplet[1], triplet[2])
        idx_sorting_fn = lambda tuplet: (tuplet[0][0], tuplet[0][1], tuplet[0][2])
        li = []
        for i in range(len(fixed)):
            li.append([fixed[i], i])
        li = sorted(li, key=idx_sorting_fn)
        sort_index = []
        for x in li:
            sort_index.append(x[1])
        label = sorted(label, key=sorting_fn)
        temp_pred = []
        temp_fixed = []
        for idx in sort_index:
            temp_pred.append(pred[idx])
            temp_fixed.append(fixed[idx])
        pred = temp_pred
        fixed = temp_fixed

        memory = set()
        memory_label = set()

        # First: match all corresponding element...
        for i, elem in enumerate(label):
            found = False
            counter = 0
            while counter < len(fixed) and not found:
                if elem == fixed[counter] and counter not in memory:
                    found = True
                    matching_comments[i] = "Found Exact"
                counter += 1

            if found:
                counter -= 1
                memory.add(counter)
                memory_label.add(i)
                new_fixed[i] = fixed[counter]
                new_pred[i] = pred[counter]
                new_label[i] = elem

        # Second matching algo: only different 1 component...
        for i, elem in enumerate(label):
            if (
                i in memory_label
            ):  # making sure not overlap from previous matching approach
                continue

            found = False
            counter = 0

            while counter < len(fixed) and not found:
                if counter not in memory:
                    is_diff_aspect = fixed[counter][0] != elem[0]
                    is_diff_sentiment = fixed[counter][1] != elem[1]
                    is_diff_polarity = fixed[counter][2] != elem[2]

                    if (
                        is_diff_aspect
                        and not is_diff_sentiment
                        and not is_diff_polarity
                    ):
                        matching_comments[i] = "Different Aspect Term"
                        found = True
                    elif (
                        not is_diff_aspect
                        and is_diff_sentiment
                        and not is_diff_polarity
                    ):
                        matching_comments[i] = "Different Sentiment Term"
                        found = True
                    elif (
                        not is_diff_aspect
                        and not is_diff_sentiment
                        and is_diff_polarity
                    ):
                        matching_comments[i] = "Different Polarity"
                        found = True
                counter += 1

            if found:
                counter -= 1
                memory.add(counter)
                memory_label.add(i)
                new_fixed[i] = fixed[counter]
                new_pred[i] = pred[counter]
                new_label[i] = elem

        # Third matching algo: each component contains label component
        for i, elem in enumerate(label):
            if (
                i in memory_label
            ):  # making sure not overlap from previous matching approach
                continue

            found = False
            counter = 0

            while counter < len(fixed) and not found:
                if counter not in memory:
                    match = True
                    aspect, sentiment, polarity = (
                        fixed[counter][0],
                        fixed[counter][1],
                        fixed[counter][2],
                    )

                    if len(aspect) <= len(elem[0]):
                        match = (aspect in elem) and match
                    else:
                        match = (elem[0] in aspect) and match

                    if len(sentiment) <= len(elem[1]):
                        match = (sentiment in elem) and match
                    else:
                        match = (elem[1] in sentiment) and match

                    match = polarity == elem[2]

                    if match:
                        found = True
                        matching_comments[i] = "Subset aspect/sentiment"
                counter += 1

            if found:
                counter -= 1
                memory.add(counter)
                memory_label.add(i)
                new_fixed[i] = fixed[counter]
                new_pred[i] = pred[counter]
                new_label[i] = elem

        # Lastly, for remaining that are not matched
        for i in range(len(label)):
            if i in memory_label:
                continue
            new_label[i] = label[i]
            matching_comments[i] = "Not Generated"

        counter = 0
        last_memory_len = len(memory)
        for j in range(len(fixed)):
            if j not in memory:
                new_fixed[counter + last_memory_len] = fixed[j]
                new_pred[counter + last_memory_len] = pred[j]
                matching_comments[counter + last_memory_len] = "No Label Matched"
                memory.add(j)
                counter += 1

        return new_label, new_pred, new_fixed, matching_comments
