import matplotlib.pyplot as plt

from utils.metrics import _agg_metric
from utils.constants import COLORS


def plot_loss(    
    # losses can be list or list of lists
    losses: list,
    labels: list = None,
    title: str = 'Loss over iterations',
    xlabel: str = 'Iteration',
    ylabel: str = 'Loss',
    save_path: str = None,
    log_scale: bool = False,
):
    fig, ax = plt.subplots()
    if not isinstance(losses[0], list):
        losses = [losses]
    for i, loss in enumerate(losses):
        ax.plot(loss, label=labels[i] if labels else f'Loss {i+1}')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    ax.legend()
    ax.grid(True)
    if save_path:
        plt.savefig(save_path)  
    plt.show()

def _rename(name, rename_dict):
    """
    Rename a metric name based on a provided dictionary.
    Args:
        name (str): Original metric name.
        rename_dict (dict): Dictionary mapping original names to new names.
    Returns:
        str: Renamed metric name.
    """
    return rename_dict.get(name, name) if rename_dict else name

def plot_metrics(
        metrics, 
        fill_between=True, 
        rename=None, 
        xlabel=r'Layer Index',
        ylabel=r'Metric Value',
        title='Metrics per Layer'
):  
    """
    Plot aggregated metrics per layer.
    Args:
        metrics (dict): Dictionary containing metric values for each layer.
        fill_between (bool): Whether to fill the area between mean and std.
        rename (dict): Optional dictionary to rename labels of metrics.
    """
    agg_metrics = {}
    for metric in metrics:
        agg_metrics[metric] = _agg_metric(metrics, metric)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric_name, (mean, std)) in enumerate(agg_metrics.items()):
        if fill_between:
            ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2, color=COLORS[i % len(COLORS)])
        ax.plot(mean, label=_rename(metric_name, rename), color=COLORS[i % len(COLORS)], marker='o')
    ax.set_xticks(range(len(mean)))
    ax.set_xticklabels([f'Layer {i}' for i in range(len(mean))], rotation=45)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend()
    plt.show()