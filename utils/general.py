import torch
import torch.nn.functional as F

import torch
from typing import Optional

def get_whole(
    prompt: str,
    llm: torch.nn.Module,
    tokenizer,
    layer_idx: int,
    input_ids: Optional[torch.Tensor] = None,
    grad: bool = False
) -> torch.Tensor:
    """
    Tokenize `prompt`, do a forward pass with output_hidden_states=True,
    and return the hidden vector of the *last* token at layer `layer_idx`.

    Args:
        prompt (str): the input string, e.g. "Harry".
        llm (nn.Module): a HuggingFace‐style model (with embeddings + hidden_states).
        tokenizer: the corresponding tokenizer for `llm`.
        layer_idx (int): which hidden‐layer index to extract (0=embeddings, 1=first block, etc.)

    Returns:
        Tensor of shape (hidden_size,) = the last‐token hidden state at `layer_idx`,
        computed under torch.no_grad().
    """
    # 1) Tokenize (and move to same device as the model).
    device = next(llm.parameters()).device
    if input_ids is None:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)      # shape (1, seq_len)
    
    if not grad:
        # Forward under no_grad and detach before returning
        with torch.no_grad():
            outputs = llm(
                input_ids=input_ids,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states
            h = hidden_states[layer_idx][0]  # shape = (seq_len, hidden_size)

        return h.detach()
    else:
        # Forward normally, so gradients can flow
        outputs = llm(
            input_ids=input_ids,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states
        h = hidden_states[layer_idx][0]  # shape = (seq_len, hidden_size)
        return h


def compute_last_token_embedding_grad(
    y: torch.LongTensor,
    llm: torch.nn.Module,
    layer_idx: int,
    h_target: torch.Tensor,
    tokenizer: Optional[torch.nn.Module],
):
    device = next(llm.parameters()).device
    y = y.to(device)
    h_target = h_target.to(device)

    emb_layer = llm.get_input_embeddings()
    if not emb_layer.weight.requires_grad:
        emb_layer.weight.requires_grad_(True)

    llm.zero_grad()
    emb_layer.zero_grad()

    with torch.set_grad_enabled(True):
        h_last = get_whole('', llm, tokenizer, layer_idx, y.unsqueeze(0), grad=True)[-1]
        diff = h_last - h_target
        loss = torch.dot(diff, diff)
        loss.backward()

    last_token_id = y[-1].item()
    grad_last_embedding = emb_layer.weight.grad[last_token_id].detach().clone() # TODO: Is this gradient with respect to the input or not?

    llm.zero_grad()
    emb_layer.zero_grad()

    return grad_last_embedding, loss.item()

def compute_all_token_embeddings_grad(
    y: torch.LongTensor,
    llm: torch.nn.Module,
    layer_idx: int,
    h_target: torch.Tensor,
    tokenizer: Optional[torch.nn.Module],
):
    """
    Compute gradients for all tokens in the sequence at once.

    Args:
        y (torch.LongTensor): Token IDs for the sequence.
        llm (torch.nn.Module): The language model.
        layer_idx (int): The layer index to target for inversion.
        h_target (torch.Tensor): Target hidden states for the sequence.
        tokenizer (Optional[torch.nn.Module]): The tokenizer for the model.

    Returns:
        torch.Tensor: Gradients for the sequence ([seq_len, hidden_size]).
        float: Total loss.
    """
    device = next(llm.parameters()).device
    y = y.to(device)
    h_target = h_target.to(device)

    emb_layer = llm.get_input_embeddings()
    if not emb_layer.weight.requires_grad:
        emb_layer.weight.requires_grad_(True)

    llm.zero_grad()
    emb_layer.zero_grad()

    with torch.set_grad_enabled(True):
        h_all = get_whole('', llm, tokenizer, layer_idx, y.unsqueeze(0), grad=True)
        diff = h_all - h_target
        loss = torch.einsum('ij,ij->', diff, diff)  # sum over seq_len
        loss.backward()

    # Extract gradients for the sequence tokens only
    grad_sequence = emb_layer.weight.grad[y].detach().clone()  # shape ([seq_len, hidden_size])

    llm.zero_grad()
    emb_layer.zero_grad()

    return grad_sequence, loss.item()
