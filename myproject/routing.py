"""Physical-head W_O V magnitudes and prediction-row source/history routing."""

from contextlib import contextmanager

import torch


def projection_grams(model):
    grams = []
    for layer in model.model.layers:
        attention = layer.self_attn
        heads = attention.q_proj.out_features // attention.head_dim
        projection = attention.o_proj.weight.detach().float()
        matrices = projection.reshape(projection.shape[0], heads, attention.head_dim)
        matrices = matrices.permute(1, 2, 0)
        grams.append(matrices @ matrices.transpose(-1, -2))
    return grams


def append_value_energy(values, gram, previous):
    """Reuse old keys' W_O V energies exactly as in the original routing collector."""
    start = 0 if previous is None else previous.shape[-1]
    unseen = values[:, start:]
    energy = ((unseen @ gram) * unseen).sum(-1).clamp_min(0)
    return energy if previous is None else torch.cat((previous, energy), dim=-1)


def route_magnitudes(attention, energy, prompt_length, source_mask, first_query):
    """Retain each physical head's ||W_O,h V_h,k|| before aggregating routing."""
    magnitude = attention * energy.sqrt()
    source = torch.zeros(attention.shape[-1], device=attention.device)
    source[:prompt_length] = torch.tensor(source_mask, device=attention.device)
    history = (torch.arange(attention.shape[-1], device=attention.device) >= prompt_length).float()
    source_norm = magnitude @ source
    history_norm = magnitude @ history
    if first_query and source_mask[-1]:
        source_norm[0] -= magnitude[0, :, prompt_length - 1]
    return dict(
        norm_source=source_norm.cpu().numpy(),
        norm_history=history_norm.cpu().numpy(),
        norm_total=magnitude.sum(-1).cpu().numpy(),
    )


@contextmanager
def observe_route(model, cache, grams, energies, prompt_length, source_mask, start):
    """Reduce each layer's current attention block; never retain all-layer edges."""
    layers = {}
    handles = []

    def observe(index):
        def record(module, args, output):
            attention = output[1][0].detach().float().transpose(0, 1)
            values = cache.layers[index].values[0].float()
            values = values.repeat_interleave(attention.shape[1] // values.shape[0], dim=0)
            energies[index] = append_value_energy(values, grams[index], energies.get(index))
            layers[index] = route_magnitudes(
                attention, energies[index], prompt_length, source_mask, start == 0,
            )
        return record

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(observe(index)))
    try:
        yield layers
    finally:
        for handle in handles:
            handle.remove()
