import matplotlib.pyplot as plt

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