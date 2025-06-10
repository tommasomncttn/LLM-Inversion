
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import BERTScorer


def _add_metric_result(metric_dict, metric_name, value):
    if metric_name not in metric_dict:
        metric_dict[metric_name] = []
    metric_dict[metric_name].append(value)

def _compute_average_metric(metric_dict, metric_name):
    if metric_name in metric_dict and len(metric_dict[metric_name]) > 0:
        return np.nanmean(metric_dict[metric_name])
    return 0.0

def compute_metrics(target_sentences, output_sentences, output_embeddings, len_treshold=3,
    rouge = Rouge(),
    bert_scorer = BERTScorer(lang='en', rescale_with_baseline=True),
):
    """
    Compute different metrics to evaluate simmilarity between predictions and ground truth
    sentences.
    Args:
        predictions (list of str): List of predicted sentences.
        ground_truth (list of str): List of ground truth sentences.
    Returns:
        dict: A dictionary containing the computed metrics.
    """
    # bleu, rouge, bertscore, ...
    metrics = {}
    for i in range(len(target_sentences)):
        if len(output_sentences[i]) < len_treshold or len(target_sentences[i]) < len_treshold:
            continue
        _add_metric_result(metrics, 'bleu', sentence_bleu(
            [output_sentences[i].split()],
            target_sentences[i].split()
        ))
        P, R, F1 = bert_scorer.score([output_sentences[i]], [target_sentences[i]])
        _add_metric_result(metrics, 'bertscore_f1', F1.item())
        try:
            rouge_s = rouge.get_scores(
                output_sentences[i], target_sentences[i]
            )[0]
        except Exception as e:
            print(f"ERROR computing ROUGE for pair {i}: {e}")
            print(f"Output: {output_sentences[i]}")
            print(f"Target: {target_sentences[i]}")
            print(len(output_sentences[i]), len(target_sentences[i]))
            rouge_s = {'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}
        _add_metric_result(metrics, 'rouge-l_f', rouge_s['rouge-l']['f'])
    for metric in metrics:
        metrics[metric] = _compute_average_metric(metrics, metric)
    return metrics
    