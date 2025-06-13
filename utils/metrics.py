
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
        np.nanmean(metric_dict[metric_name], axis=1),
        np.nanstd(metric_dict[metric_name], axis=1)
    )
    return res

def _agg_metric_over_time(metric_dict, metric_name, window_size=1):
    """
    Aggregate metrics over time.
    Args:
        metrics (dict): Dictionary containing metric values over time.
    Returns:
        dict: Average and compute the std of all points smaller that each time point.
    """
    # {metric}_time should be in the metric_dict
    time_name = f'{metric_name}_time'
    assert time_name in metric_dict, f"Metrics must contain '{time_name}' key for aggregation of '{metric_name}'."
    assert metric_name in metric_dict, f"Metric '{metric_name}' not found in metric_dict."

    times_metric = np.array(list(zip(metric_dict[time_name].flatten(), metric_dict[metric_name].flatten())))
    sorted_indices = np.argsort(times_metric[:, 0])
    sorted_times_metric = times_metric[sorted_indices]

    agg = []
    values = sorted_times_metric[:, 1]
    for i in range(len(sorted_times_metric)):
        time = sorted_times_metric[i, 0]
        if window_size is not None and i < window_size:
            continue
        if window_size is not None:
            values_before = values[i - window_size:i + 1]
        else:
            values_before = values[:i + 1]
        mean = np.nanmean(values_before)
        std = np.nanstd(values_before)
        agg.append((time, mean, std))
    return agg


def compute_metrics(target_sentences, sentense_embeddings, output_sentences, output_embeddings, len_treshold=2,
    rouge = Rouge(),
    bert_scorer = BERTScorer(lang='en', rescale_with_baseline=True),
):
    """
    Compute various metrics between target sentences and output sentences.
    Args:
        target_sentences (list of str): The ground truth sentences.
        sentense_embeddings (list of torch.Tensor): The embeddings for the ground truth sentences.
        output_sentences (list of str): The generated output sentences.
        output_embeddings (list of torch.Tensor): The embeddings for the output sentences.
        len_treshold (int): Minimum length threshold for considering a sentence pair.
        rouge (Rouge): Rouge metric object.
        bert_scorer (BERTScorer): BERTScore metric object.
    Returns:
        dict: A dictionary containing computed metrics.
    """
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
        _add_metric_result(metrics, 'l2_distance', np.linalg.norm(
            sentense_embeddings[i] - output_embeddings[i],
            axis=1
        ).mean())
    if len(target_sentences) == 1:
        for key in metrics:
            metrics[key] = metrics[key][0]
    return metrics
    