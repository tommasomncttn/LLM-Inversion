
import torch
from tqdm import tqdm

from utils.general import compute_all_token_embeddings_grad, extract_hidden_states_ids

def invert_whole_prompt(prompt, model, tokenizer, layer_idx, n_iterations=1000, gamma=1e-1):
    """
    Invert the entire prompt at once by optimizing embeddings for all tokens simultaneously.

    Args:
        prompt (str): The input prompt to invert.
        model: The language model.
        tokenizer: The tokenizer for the model.
        layer_idx (int): The layer index to target for inversion.
        n_iterations (int): Number of optimization iterations.
        gamma (float): Step size for gradient descent.

    Returns:
        str: The reconstructed prompt.
    """
    # Tokenize the prompt and get target hidden states
    tokenized = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized["input_ids"].squeeze(0)
    h_target = extract_hidden_states_ids(
        input_ids=input_ids.unsqueeze(0),
        llm=model,
        layer_idx=layer_idx,
        grad=False
    )
    # Initialize random embeddings for the entire sequence
    embedding_matrix = model.get_input_embeddings().weight
    vocab_size, hidden_size = embedding_matrix.shape
    random_ids = torch.randint(0, vocab_size, (input_ids.size(0),))
    x_i_plus_1 = embedding_matrix[random_ids]
    losses = []
    distances = []

    with tqdm(total=n_iterations, desc="Inverting prompt") as pbar:
        for iteration in range(n_iterations):
            # Compute gradients for the entire sequence
            grad_oracle, loss = compute_all_token_embeddings_grad(
                y=random_ids,
                llm=model,
                layer_idx=layer_idx,
                h_target=h_target,
                tokenizer=tokenizer,
            ) 
            losses.append(loss)
            x_i_plus_1 = x_i_plus_1 - gamma * grad_oracle

            dist = torch.cdist(x_i_plus_1, embedding_matrix)
            random_ids = torch.argmin(dist, dim=1)

            dist_from_prompt = torch.norm(
                embedding_matrix[random_ids] - embedding_matrix[input_ids]
            )
            average_distance = dist_from_prompt.mean().item()
            distances.append(average_distance)

            pbar.set_postfix({"Loss": loss, "Distance": average_distance})
            pbar.update(1)

            if average_distance < 1e-3:
                break

    reconstructed_prompt = tokenizer.decode(random_ids.tolist(), skip_special_tokens=True)
    return reconstructed_prompt, losses, distances