
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import BERTScorer


def _add_metric_result(metric_dict, metric_name, value):
    if metric_name not in metric_dict:
        metric_dict[metric_name] = []
    metric_dict[metric_name].append(value)

def _agg_metric(metric_dict, metric_name):
    """
    Aggregate metric values from the metric dictionary.
    Args:
        metric_dict (dict): Dictionary containing metric values.
        metric_name (str): Name of the metric to aggregate.
    Returns:
        tuple: mean and std 
    """
    if metric_name not in metric_dict:
        raise ValueError(f"Metric '{metric_name}' not found in metric_dict.")
    res = (
        np.mean(metric_dict[metric_name], axis=1),
        np.std(metric_dict[metric_name], axis=1)
    )
    return res

def compute_metrics(target_sentences, sentense_embeddings, output_sentences, output_embeddings, len_treshold=3,
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
        # add l2 distance between output and target embeddings
        _add_metric_result(metrics, 'l2_distance', np.linalg.norm(
            sentense_embeddings[i] - output_embeddings[i]
        ))
        
    return metrics
    