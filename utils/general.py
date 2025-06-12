import random
import numpy as np
import torch
import torch.nn.functional as F

import torch
from typing import Optional, Tuple

def extract_hidden_states_prompt(
    prompt: str,
    llm: torch.nn.Module,
    tokenizer,
    layer_idx: int,
    grad: bool = False,
) -> torch.Tensor:
    """
    Tokenize `prompt`, do a forward pass with output_hidden_states=True,
    and return the hidden vector of the *last* token at layer `layer_idx`.
    Args:
        prompt (str): the input string, e.g. "Harry".
        llm (nn.Module): a HuggingFace‐style model (with embeddings + hidden_states).
        tokenizer: the corresponding tokenizer for `llm`.
        layer_idx (int): which hidden‐layer index to extract (0=embeddings, 1=first block, etc.)
        grad (bool): whether to compute gradients or not.
    Returns:
        Tensor of shape (hidden_size,) = the last‐token hidden state at `layer_idx`,
        computed under torch.no_grad() if grad=False, otherwise gradients are enabled.
    """
    device = next(llm.parameters()).device
    # if input_ids is None:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)      # shape (1, seq_len)
    return extract_hidden_states_ids(
        input_ids=input_ids,
        llm=llm,
        layer_idx=layer_idx,
        grad=grad
    )

def extract_hidden_states_ids(
    input_ids: torch.LongTensor,
    llm: torch.nn.Module,
    layer_idx: int,
    grad: bool = False
) -> torch.Tensor:
    """
    Extract hidden states for the last token in a sequence of input IDs.
    Args:
        input_ids (torch.LongTensor): Token IDs for the sequence.
        llm (torch.nn.Module): The language model.
        layer_idx (int): The layer index to target for inversion.
        grad (bool): Whether to compute gradients or not.
    Returns:
        torch.Tensor: Hidden state of the last token at the specified layer index.
    """
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
        h_last = extract_hidden_states_prompt('', llm, tokenizer, layer_idx, y.unsqueeze(0), grad=True)[-1]
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
        h_all = extract_hidden_states_prompt('', llm, tokenizer, layer_idx, y.unsqueeze(0), grad=True)
        diff = h_all - h_target
        loss = torch.einsum('ij,ij->', diff, diff)  # sum over seq_len
        loss.backward()

    # Extract gradients for the sequence tokens only
    grad_sequence = emb_layer.weight.grad[y].detach().clone()  # shape ([seq_len, hidden_size])

    llm.zero_grad()
    emb_layer.zero_grad()

    return grad_sequence, loss.item()


def extract_hidden_states(
    embeddings: torch.Tensor,
    llm: torch.nn.Module,
    layer_idx: int,
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
    if not grad:
        # Forward under no_grad and detach before returning
        with torch.no_grad():
            outputs = llm(
                inputs_embeds=embeddings,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states
            h = hidden_states[layer_idx][0]  # shape = (seq_len, hidden_size)

        return h.detach()
    else:
        # Forward normally, so gradients can flow
        outputs = llm(
            inputs_embeds=embeddings,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states
        h = hidden_states[layer_idx][0]  # shape = (seq_len, hidden_size)
        return h

def compute_last_token_embedding_grad_emb(
    embeddings: torch.Tensor,
    llm: torch.nn.Module,
    layer_idx: int,
    h_target: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """
    Given a batch of precomputed token embeddings, run a forward pass
    up to `layer_idx`, compute the MSE loss against `h_target` for the last token,
    and return the gradient w.r.t. that last-token embedding plus the loss value.

    Gradients are computed only for the last embedding row; the others are treated as constants.

    Args:
        embeddings:  Tensor of shape (1, seq_len, hidden_size)
        llm:         A HuggingFace-style model supporting inputs_embeds + hidden_states.
        layer_idx:   Index of the hidden layer to extract (0=embeddings, 1=first block, ...).
        h_target:    Tensor of shape (hidden_size,) giving the desired hidden state for the last token.

    Returns:
        grad_last_embedding: Tensor of shape (hidden_size,) = d(loss)/d(embeddings[0,-1,:]).
        loss_val:            Scalar float = loss.item().
    """
    # Move to device
    device = next(llm.parameters()).device
    embeddings = embeddings.to(device)
    h_target = h_target.to(device)

    # Split out the last-token embedding as a separate tensor requiring grad
    fixed_embs = embeddings.clone().detach()
    last_emb = fixed_embs[:, -1:, :].clone().requires_grad_(True)

    # Reassemble inputs_embeds with fixed prefixes and grad-enabled last
    inputs_embeds = torch.cat([fixed_embs[:, :-1, :], last_emb], dim=1)

    # Forward pass from custom embeddings
    outputs = llm(
        inputs_embeds=inputs_embeds,
        output_hidden_states=True
    )
    hidden_states = outputs.hidden_states
    h_last = hidden_states[layer_idx][0, -1, :]

    # Compute MSE loss for last token
    loss = torch.nn.functional.mse_loss(h_last, h_target, reduction='sum')

    # Compute gradient only w.r.t. last_emb
    
    loss.backward()
    return last_emb.grad.squeeze(0, 1), loss

    # grad_last = torch.autograd.grad(loss, last_emb)[0]  # shape (1,1,hidden_size)
    # grad_last_embedding = grad_last[0, 0, :].detach().clone().requires_grad_(False)
    # return grad_last_embedding, loss


def set_seed(seed: int = 8):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import sys
import importlib
import os
from os.path import splitext, basename, dirname, isfile

def load_module(module_reference, name):
    # Check if it's a file path to a .py file
    if isfile(module_reference) and module_reference.endswith('.py'):
        mod_name = splitext(basename(module_reference))[0]
        mod_path = dirname(os.path.abspath(module_reference))
        sys.path.insert(0, mod_path)
        module = __import__(mod_name)
    else:
        # Assume it's an installed module
        module = importlib.import_module(module_reference)

    return getattr(module, name)

def load_model(src, cls, args=None):
    model = load_module(src, cls)
    return model(**args) if args else model()