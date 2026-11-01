"""Released-formula LUMINA tensor core and full-prefix dense observation.

Adapted from deeplearning-wisc/LUMINA, commit
c43ff41d872b05f659dcb3ad3a6dd78226954319 (lumina.py SHA256
1b5ecd8c08982b40dae59f299e580219074ee83673768feb53cf15612b7d8a5f).
Includes the released final-state/norm convention, not the paper's L-1 variant.
"""

# MIT License. Copyright (c) 2025 Min-Hsuan, Yeh
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from myproject.inputs import encode_response, encode_source
from myproject.model import attention_backend, input_ids


CHUNK = 64
LAYER_BLOCK = 4
TOPK = 100
CORE = {
    'repository': 'https://github.com/deeplearning-wisc/LUMINA',
    'revision': 'c43ff41d872b05f659dcb3ad3a6dd78226954319',
    'upstream_sha256': '1b5ecd8c08982b40dae59f299e580219074ee83673768feb53cf15612b7d8a5f',
    'definition': 'released-code all hidden_states[1:], including final norm reapplied',
    'lambda': .5, 'topk': TOPK, 'kernel': '(1+cosine)/2',
    'topk_renormalized': False, 'mmd': 'squared; unclipped',
    'precision': 'native softmax/kernel; released reductions; computed kernel cast float32 for BF16 weight compatibility',
    'bf16_compatibility_fix': 'upstream cosine result lacks float32 cast required by its float32 top100 weights',
    'chunk': CHUNK, 'layer_block': LAYER_BLOCK, 'fit': False, 'judge': False,
    'backbone_execution': 'dense_full_prefix', 'prefix_cache_reuse': False,
}


def _finite(*values):
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError('LUMINA requires finite tensors and outputs')


@torch.no_grad()
def topk_view(probability, embedding):
    """Retain original probability mass and the model's input embedding rows."""
    if probability.ndim != 2 or not len(probability) or probability.shape[1] < TOPK:
        raise ValueError('LUMINA requires nonempty token rows and at least100 vocabulary entries')
    _finite(probability)
    if torch.any(probability < 0):
        raise ValueError('LUMINA probabilities must be nonnegative')
    weights, ids = torch.topk(probability, TOPK, dim=-1)
    return weights.float(), embedding(ids)


def _cosine(x, y):
    x, y = F.normalize(x, p=2, dim=-1, eps=1e-8), F.normalize(y, p=2, dim=-1, eps=1e-8)
    return ((1+torch.matmul(x, y.T))/2).float()


@torch.no_grad()
def mmd_from_topk(p_weights, p_embed, q_weights, q_embed):
    """Unnormalized top100 squared MMD, preserving the released operation order."""
    if (p_weights.ndim != 2 or p_weights.shape != q_weights.shape
            or p_weights.shape[1] != TOPK or not len(p_weights)
            or p_embed.shape != q_embed.shape or p_embed.shape[:2] != p_weights.shape
            or p_embed.ndim != 3):
        raise ValueError('Matching nonempty top100 token and embedding axes required')
    _finite(p_weights, q_weights, p_embed, q_embed)
    result = []
    for first in range(0, len(p_weights), 32):
        rows = range(first, min(first+32, len(p_weights)))
        pp = torch.stack([p_weights[t] @ _cosine(p_embed[t], p_embed[t]) @ p_weights[t] for t in rows])
        qq = torch.stack([q_weights[t] @ _cosine(q_embed[t], q_embed[t]) @ q_weights[t] for t in rows])
        pq = torch.stack([p_weights[t] @ _cosine(p_embed[t], q_embed[t]) @ q_weights[t] for t in rows])
        result.append(pp+qq-2*pq)
    result = torch.cat(result).cpu()
    _finite(result)
    return result


@torch.no_grad()
def context_mmd(p_probability, q_probability, embedding):
    """Public full-probability core, also used by the upstream equivalence tests."""
    if p_probability.shape != q_probability.shape:
        raise ValueError('Source and random probabilities must share token/vocabulary axes')
    p_weights, p_embed = topk_view(p_probability, embedding)
    q_weights, q_embed = topk_view(q_probability, embedding)
    return mmd_from_topk(p_weights, p_embed, q_weights, q_embed)


@torch.no_grad()
def layer_statistics(probability, final_argmax):
    """Reduce full-vocabulary layer blocks using native per-row arithmetic."""
    entropy = -torch.sum(probability*torch.log(probability.clamp(min=1e-8)), dim=-1)
    indices = final_argmax.expand(probability.shape[:-1]).unsqueeze(-1)
    return entropy, probability.gather(-1, indices).squeeze(-1)


@torch.no_grad()
def information_rate(entropy, layer_max_probability, final_probability, targets):
    """Same released IPR from sufficient J×T statistics; no fitted calibration."""
    if (entropy.ndim != 2 or entropy.shape != layer_max_probability.shape or not entropy.numel()
            or final_probability.ndim != 2 or final_probability.shape[0] != entropy.shape[1]
            or targets.shape != (entropy.shape[1],)):
        raise ValueError('Matching layer/token/vocabulary and target axes required')
    _finite(entropy, layer_max_probability, final_probability)
    maximum = final_probability.max(dim=-1).values
    if torch.any(maximum <= 0) or torch.any(entropy < 0):
        raise ValueError('Positive final probabilities and nonnegative entropy required')
    layers = torch.arange(1, len(entropy)+1, device=entropy.device).unsqueeze(1)
    weights = 1.0/(entropy+1e-8)
    ratios = 1-torch.clamp(layer_max_probability/maximum.unsqueeze(0), max=1.0)
    numerator = (ratios*layers).sum(dim=0)
    denominator = (layers*weights).sum(dim=0)
    observed = final_probability.gather(1, targets[:, None])[:, 0]
    result = ((numerator/denominator)*(observed/maximum)).cpu()
    _finite(result)
    return result


def random_prompt(prompt, source, donor):
    """Replace exactly one evidence substring, keeping the query and instructions."""
    if not source or not donor or prompt.count(source) != 1 or source == donor:
        raise ValueError('Distinct nonempty donor and unique original source required')
    return prompt.replace(source, donor, 1)


@torch.no_grad()
def _present(model, output, targets):
    logits = model.lm_head(output.last_hidden_state[0])
    probability = F.softmax(logits, dim=-1).float()
    maximum = probability.argmax(dim=-1)
    norm = model.model.norm
    hidden = output.hidden_states[1:]
    stats = []
    for first in range(0, len(hidden), LAYER_BLOCK):
        block = torch.stack([row[0] for row in hidden[first:first+LAYER_BLOCK]])
        probability_block = F.softmax(model.lm_head(norm(block)), dim=-1)
        stats.append(layer_statistics(probability_block, maximum))
    entropy, layer_max = (torch.cat(values) for values in zip(*stats, strict=True))
    ipr = information_rate(entropy, layer_max, probability, targets)
    top = topk_view(probability, model.get_input_embeddings())
    logp = logits.float().gather(1, targets[:, None])[:, 0]-logits.float().logsumexp(-1)
    return ipr, top, logp.cpu()


@torch.no_grad()
def measure_group(model, tokenizer, prompt, source, donor, responses, *, chunk=CHUNK):
    """Three dense S/R/A forwards per response; all original answer IDs retained."""
    if type(chunk) is not int or chunk < 1 or not responses:
        raise ValueError('Positive query chunk and nonempty responses required')
    changed = random_prompt(prompt, source, donor)
    original = encode_source(tokenizer, prompt, source)
    present, absent = original['prompt_with_source'], original['prompt_without_source']
    random = encode_source(tokenizer, changed, donor)['prompt_with_source']
    prefixes = (present, random, absent)
    encoded = [encode_response(tokenizer, response) for response in responses]
    limit = getattr(model.config, 'max_position_embeddings', None)
    for row in encoded:
        if not row['answer_ids']:
            raise ValueError('Original response must contain tokens')
        if isinstance(limit, int) and max(map(len, prefixes))+len(row['answer_ids']) > limit:
            raise ValueError('All complete LUMINA S/R/A worlds must fit the context budget; truncation is forbidden')
    results = []
    with attention_backend(model, 'sdpa'):
        for row in encoded:
            answer = row['answer_ids']
            worlds = [model.model(input_ids=input_ids(model, [*prefix, *answer]),
                                  use_cache=False, output_hidden_states=index == 0, return_dict=True)
                      for index, prefix in enumerate(prefixes)]
            chunks = {name: [] for name in ('lumina', 'mmd', 'ipr', 'present_logp', 'random_logp', 'absent_logp')}
            for first in range(0, len(answer), chunk):
                stop = min(first+chunk, len(answer))
                targets = input_ids(model, answer[first:stop])[0]
                rows = slice(len(present)-1+first, len(present)-1+stop)
                view = SimpleNamespace(last_hidden_state=worlds[0].last_hidden_state[:, rows],
                    hidden_states=tuple(hidden[:, rows] for hidden in worlds[0].hidden_states))
                ipr, p_top, p_logp = _present(model, view, targets)
                logits = [model.lm_head(world.last_hidden_state[0, len(prefix)-1+first:len(prefix)-1+stop])
                          for prefix, world in zip(prefixes[1:], worlds[1:], strict=True)]
                q_top = topk_view(F.softmax(logits[0], dim=-1).float(), model.get_input_embeddings())
                mmd = mmd_from_topk(*p_top, *q_top)
                logp = [(values.float().gather(1, targets[:, None])[:, 0]-values.float().logsumexp(-1)).cpu()
                        for values in logits]
                for name, values in zip(chunks, (.5*ipr-.5*mmd, mmd, ipr, p_logp, *logp), strict=True):
                    _finite(values)
                    chunks[name].append(values.numpy().astype(np.float64))
                del view, logits, p_top, q_top
            results.append({name: np.concatenate(values) for name, values in chunks.items()})
            del worlds
    return results, encoded
