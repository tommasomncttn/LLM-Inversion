from time import time
import torch
from torch.optim import Adam
from tqdm import tqdm

from utils.general import extract_hidden_states_ids # Assuming this can handle inputs_embeds

def invert_whole_prompt(
    prompt,
    model,
    tokenizer,
    layer_idx,
    n_iterations=2000,
    lr=0.1
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
    device = model.device
    tokenized = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    
    with torch.no_grad():
        # Target hidden states should not track gradients
        h_target = model(input_ids, output_hidden_states=True).hidden_states[layer_idx]

    embedding_matrix = model.get_input_embeddings().weight
    vocab_size, hidden_size = embedding_matrix.shape
    
    random_ids = torch.randint(0, vocab_size, (1, input_ids.size(1)), device=device)
    # Clone and set requires_grad=True to make it a leaf tensor for optimization
    optimized_embeddings = embedding_matrix[random_ids].clone().detach().requires_grad_(True)

    optimizer = Adam([optimized_embeddings], lr=lr)
    losses, times, steps, l2s = [], [], [], []
    start_time = time()

    with tqdm(total=n_iterations, desc="Inverting prompt") as pbar:
        for iteration in range(n_iterations):
            current_outputs = model(inputs_embeds=optimized_embeddings, output_hidden_states=True)
            h_current = current_outputs.hidden_states[layer_idx]
            loss = torch.nn.functional.mse_loss(h_current, h_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                dist = torch.cdist(optimized_embeddings.squeeze(0), embedding_matrix)
                projected_ids = torch.argmin(dist, dim=1)
                # Calculate the distance between continuous optimized embeds and discrete projected embeds
                projected_embeddings = embedding_matrix[projected_ids]
                l2 = torch.norm(
                    optimized_embeddings.squeeze(0) - projected_embeddings, dim=1
                ).mean().item()

            losses.append(loss.item())
            times.append(time() - start_time)
            steps.append(iteration)
            l2s.append(l2)

            pbar.set_postfix({"Loss": loss.item(), "L2": l2})
            pbar.update(1)

            if torch.equal(projected_ids, input_ids.squeeze(0)):
                print("\nConverged to original prompt!")
                break
    
    reconstructed_prompt = tokenizer.decode(projected_ids.tolist(), skip_special_tokens=True)
    return reconstructed_prompt, losses, times, steps, l2s