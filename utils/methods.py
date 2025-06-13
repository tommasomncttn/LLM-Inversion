from time import time
import torch
from torch.optim import Adam
from tqdm import tqdm

from utils.general import compute_last_token_embedding_grad_emb, extract_hidden_states_ids, consolidate_logs, extract_hidden_states_prompt, set_seed
from utils.metrics import compute_metrics


def test_prompt_reconstruction(model, tokenizer, sentences, seed=8, **kwargs):
    """
    Test the reconstruction of prompts using the model's hidden states.
    Args:
        model: The language model to use for reconstruction.
        tokenizer: The tokenizer corresponding to the language model.
        dataset: A list of sentences to process.
        seed: Random seed for reproducibility.
    Returns:
        A dictionary containing the computed metrics for each layer.
    """
    embedding_matrix = model.get_input_embeddings().weight
    metrics = []
    # average scores over the dataset
    sentense_embeddings = []
    for sentence in tqdm(sentences, desc="Computing sentence embeddings"):
        set_seed(seed)
        input_ids = tokenizer(sentence, return_tensors='pt').input_ids
        with torch.no_grad():
            embedding = model.get_input_embeddings()(input_ids)[0]
            embedding = embedding.numpy()
        sentense_embeddings.append(embedding)
    
    for layer_idx in range(model.config.num_hidden_layers):
        output_sentences = []
        output_embeddings = []
        for prompt in tqdm(sentences, desc="Processing prompts"):
            set_seed(seed)
            h_target = extract_hidden_states_prompt(prompt, model, tokenizer,layer_idx=layer_idx)
            output_tokens = (h_target @ embedding_matrix.T).argmax(dim=-1)
            output_sentence = tokenizer.decode(output_tokens, skip_special_tokens=True)
            output_sentences.append(output_sentence)
            output_embeddings.append(embedding_matrix[output_tokens].detach().numpy())
        metrics.append(compute_metrics(sentences, sentense_embeddings, output_sentences, output_embeddings))
    merge_metrics = consolidate_logs(metrics)
    return merge_metrics

def gd_all_tokens(
    model,
    tokenizer,
    prompt,
    layer_idx,
    n_iterations=2000,
    lr=0.1,
    log_freq=100,
    seed = 8,
    **kwargs
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
    set_seed(seed)
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
                    output_sentence = tokenizer.decode(projected_ids.tolist(), skip_special_tokens=True)
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
    merge_logs = consolidate_logs(logs)
    return reconstructed_prompt, merge_logs


def exhaustive_search(model, tokenizer, prompt, layer_idx=0, seed=42, eps=1e-3, **kwargs):
    """
    Exhaustively search for the best token at each position in the prompt
    Args:
        model: The language model to use for extraction.
        tokenizer: The tokenizer for the model.
        prompt (str): The input prompt to invert.
        layer_idx (int): The layer index to target for inversion.
        seed (int): Random seed for reproducibility.
        eps (float): Tolerance for loss convergence.
    Returns:
        tuple: (output_sentence, logs)
    """
    set_seed(seed)
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    embedding_matrix = model.get_input_embeddings().weight
    h_target = extract_hidden_states_prompt(prompt, model, tokenizer, layer_idx=layer_idx)

    output_tokens = []
    time_start = time()
    step = 0
    logs = []
    for i in range(input_ids.shape[1]):
        min_loss = float('inf')
        best_token = None
        bar = tqdm(range(embedding_matrix.shape[0]), desc=f"Finding token {i+1}/{input_ids.shape[1]}")
        # try each token in the vocabulary, compute the loss, pick the best one
        for j in bar:
            step += 1
            current_tokens = output_tokens + [j]
            current_tokens = torch.tensor(current_tokens).unsqueeze(0)
            h = extract_hidden_states_ids(current_tokens, model, layer_idx)
            loss = (h_target[i] - h[i]).norm()
            if loss < min_loss:
                min_loss = loss
                best_token = j
            bar.set_postfix({'loss': min_loss.item(), 'token': tokenizer.decode(best_token)})
            if loss < eps:
                break
        output_tokens.append(best_token)
        output_sentence = tokenizer.decode(output_tokens, skip_special_tokens=True)
        output_embedding = torch.zeros_like(h_target)
        output_embedding[:i+1] = embedding_matrix[output_tokens].detach()
        m = compute_metrics(
            [prompt], 
            [embedding_matrix[input_ids].detach().numpy()],
            [output_sentence],
            [output_embedding.numpy()],
        )
        m['step'] = step 
        m['time'] = time() - time_start 
        logs.append({k: v for k, v in m.items()})

    output_sentence = tokenizer.decode(output_tokens, skip_special_tokens=True)
    logs = consolidate_logs(logs)
    return output_sentence, logs


def find_token(
    token_idx,
    embedding_matrix,
    discovered_embeddings, discovered_ids,
    model, tokenizer, layer_idx, h_target,
    optimizer_cls, lr,
    step = 0
):
    copy_embedding_matrix = embedding_matrix.clone().detach().requires_grad_(False)

    token_id = torch.randint(0, embedding_matrix.size(0), (1,)).item()
    
    embedding = copy_embedding_matrix[token_id].clone().requires_grad_(True)
    temp_embedding = copy_embedding_matrix[token_id].clone().detach()

    optimizer = optimizer_cls([embedding], lr=lr)

    bar = tqdm(
        range(embedding_matrix.size(0)), 
        desc=f'Token [{token_idx + 1:2d}/{h_target.size(0):2d}]'
    )

    for _ in bar:
        step += 1
        input_embeddings = torch.stack(
            discovered_embeddings + [temp_embedding]
        ).unsqueeze(0) 

        grad_oracle, loss = compute_last_token_embedding_grad_emb(
            embeddings=input_embeddings, 
            model=model,
            layer_idx=layer_idx,
            h_target=h_target[token_idx],
        )

        grad_norm = grad_oracle.norm().item()
        string_so_far = tokenizer.decode(discovered_ids + [token_id], skip_special_tokens=True)
        bar.set_postfix_str(f"Loss: {loss:.2e} - Gradient norm: {grad_norm:.2e} - String: {string_so_far}")

        if loss < 1e-5 or grad_norm < 1e-12:
            break

        embedding.grad = grad_oracle
        optimizer.step()

        copy_embedding_matrix[token_id] = float('inf')
        distances = torch.norm(copy_embedding_matrix - embedding, dim=1)
        token_id = int(torch.argmin(distances))
        temp_embedding = copy_embedding_matrix[token_id].clone()

    return token_id, copy_embedding_matrix[token_id], step


# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def gd_sequential(
    model, tokenizer, prompt, layer_idx,
    optimizer_cls, lr,
    seed=8
):
    """
    Sequentially invert the prompt using a PyTorch optimizer.
    Args:
        model: The language model to use for inversion.
        tokenizer: The tokenizer for the model.
        prompt (str): The input prompt to invert.
        layer_idx (int): The layer index to target for inversion.
        optimizer_cls: The optimizer class to use (e.g., Adam).
        lr (float): Learning rate for the optimizer.
        seed (int): Random seed for reproducibility.
    Returns:
        tuple: (reconstructed_prompt, logs)
    """
    print(f"Starting inversion for prompt: {prompt}")
    set_seed(seed)
    h_target = extract_hidden_states_prompt(prompt, model, tokenizer, layer_idx)

    embedding_matrix = model.get_input_embeddings().weight
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    input_embeddings = model.get_input_embeddings()(input_ids)

    if h_target.dim() == 1:
        h_target = h_target.unsqueeze(0)

    discovered_embeddings = []
    discovered_ids        = []

    start_time = time()
    step = 0
    logs = []
    for i in range(h_target.size(0)):
        next_token_id, next_token_embedding, step = find_token(
            i, embedding_matrix,
            discovered_embeddings, discovered_ids,
            model, tokenizer, layer_idx, h_target,
            optimizer_cls, lr,
            step=step
        )
        discovered_embeddings.append(next_token_embedding)
        discovered_ids.append(next_token_id)
        discovered_embeddings_extended = torch.zeros_like(h_target)
        discovered_embeddings_extended[:i+1] = torch.stack(discovered_embeddings)
        m = compute_metrics(
            [prompt], 
            [input_embeddings.detach().numpy()],
            [tokenizer.decode(discovered_ids, skip_special_tokens=True)],
            [discovered_embeddings_extended.numpy()],
        )
        m['step'] = step
        m['time'] = time() - start_time
        logs.append({k: v for k, v in m.items()})

    final_string = tokenizer.decode(discovered_ids, skip_special_tokens=True)
    logs = consolidate_logs(logs)
    return final_string, logs


    