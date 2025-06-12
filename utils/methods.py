from time import time
import torch
from torch.optim import Adam
from tqdm import tqdm

from utils.general import extract_hidden_states_ids
from utils.metrics import compute_metrics

def invert_whole_prompt(
    prompt,
    model,
    tokenizer,
    layer_idx,
    n_iterations=2000,
    lr=0.1,
    log_freq=100
):
    """
    Invert the entire prompt using a PyTorch optimizer.

    Args:
        prompt (str): The input prompt to invert.
        model: The language model.
        tokenizer: The tokenizer for the model.
        layer_idx (int): The layer index to target for inversion.
        n_iterations (int): Number of optimization iterations.
        lr (float): Learning rate for the Adam optimizer.

    Returns:
        tuple: (reconstructed_prompt, losses, times, steps, distances)
    """
    embedding_matrix = model.get_input_embeddings().weight
    vocab_size, hidden_size = embedding_matrix.shape

    device = model.device
    tokenized = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    input_sentence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    input_embeddings = model.get_input_embeddings()(input_ids)
    
    with torch.no_grad():
        # Target hidden states should not track gradients
        h_target = model(input_ids, output_hidden_states=True).hidden_states[layer_idx]

    
    random_ids = torch.randint(0, vocab_size, (1, input_ids.size(1)), device=device)
    # Clone and set requires_grad=True to make it a leaf tensor for optimization
    optimized_embeddings = embedding_matrix[random_ids].clone().detach().requires_grad_(True)

    optimizer = Adam([optimized_embeddings], lr=lr)
    start_time = time()
    logs = []
    with tqdm(total=n_iterations, desc="Inverting prompt") as pbar:
        for iteration in range(n_iterations):
            current_outputs = model(inputs_embeds=optimized_embeddings, output_hidden_states=True)
            h_current = current_outputs.hidden_states[layer_idx]
            loss = torch.nn.functional.mse_loss(h_current, h_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if iteration % log_freq == 0 or iteration == n_iterations - 1:
                with torch.no_grad():
                    dist = torch.cdist(optimized_embeddings.squeeze(0), embedding_matrix)
                    projected_ids = torch.argmin(dist, dim=1)
                    projected_embeddings = embedding_matrix[projected_ids]
                    output_sentence = tokenizer.decode(projected_ids.tolist(), skip_special_tokens=True)
                    # print(f"input sentence: {input_sentence}")
                    # print(f"output sentence: {output_sentence}")
                    # print(f"input embeddings: {input_embeddings.squeeze(0).shape}")
                    # print(f"output embeddings: {optimized_embeddings.squeeze(0).shape}")

                    m = compute_metrics(
                        [input_sentence],
                        [input_embeddings],
                        [output_sentence],
                        [optimized_embeddings.squeeze(0)],
                    )
                    m['step'] = iteration + 1
                    m['time'] = time() - start_time
                    m['loss'] = loss.item()
                    logs.append({k:v for k, v in m.items()})

            pbar.set_postfix({"Loss": loss.item(), "L2": logs[-1]['l2_distance'].item() if logs else None})
            pbar.update(1)

            if torch.equal(projected_ids, input_ids.squeeze(0)):
                print("\nConverged to original prompt!")
                break
    
    reconstructed_prompt = tokenizer.decode(projected_ids.tolist(), skip_special_tokens=True)
    merge_logs = {}
    for log in logs:
        for key, value in log.items():
            if key not in merge_logs:
                merge_logs[key] = []
            merge_logs[key].append(value)
    return reconstructed_prompt, merge_logs