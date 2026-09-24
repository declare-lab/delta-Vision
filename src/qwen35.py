"""Qwen3.5 visual adapter, hybrid decoder integration and shared train/eval workflow."""


# Static visual embedding adapter for Qwen3.5's hybrid decoder.
from contextlib import contextmanager
import types

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def install_fast_kernels():
    """Explicit, fail-closed binding; never silently use the Python recurrence."""
    import triton
    from packaging.version import Version
    assert Version(triton.__version__) >= Version('3.7.1'), (
        'FLA gated backward on H200 requires Triton >=3.7.1; use the isolated qwen35_python dependencies')
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

    def chunk(q, k, v, **kw):
        allowed = {x: kw[x] for x in ('g', 'beta', 'initial_state', 'output_final_state',
                   'use_qk_l2norm_in_kernel', 'cu_seqlens') if x in kw}
        return chunk_gated_delta_rule(q, k, v, **allowed)

    def recurrent(q, k, v, **kw):
        allowed = {x: kw[x] for x in ('g', 'beta', 'initial_state', 'output_final_state',
                   'use_qk_l2norm_in_kernel', 'cu_seqlens') if x in kw}
        return fused_recurrent_gated_delta_rule(q, k, v, **allowed)

    def conv(x, w, bias=None, activation=None, **kw):
        return causal_conv1d_fn(x, w, bias, activation=activation)

    m.torch_chunk_gated_delta_rule = chunk
    m.torch_recurrent_gated_delta_rule = recurrent
    m.causal_conv1d_fn = conv
    m.causal_conv1d_update = causal_conv1d_update
    return {'linear_prefill': 'fla.chunk_gated_delta_rule',
            'linear_decode': 'fla.fused_recurrent_gated_delta_rule',
            'convolution': 'causal_conv1d CUDA', 'full_attention': 'flash_attention_2'}


class StaticVisualAdapter(nn.Module):
    def __init__(self, hidden_size=2560, num_layers=32, rank=128):
        super().__init__()
        self.down = nn.ModuleList(nn.Linear(hidden_size, rank, bias=False) for _ in range(num_layers))
        self.up = nn.ModuleList(nn.Linear(rank, hidden_size, bias=False) for _ in range(num_layers))
        for layer in self.up:
            nn.init.zeros_(layer.weight)

    def forward(self, embeddings):
        # FP32 optimizer/master weights; matmuls use the backbone BF16 dtype.
        with torch.autocast(device_type=embeddings.device.type, dtype=embeddings.dtype,
                            enabled=embeddings.dtype in (torch.float16, torch.bfloat16)):
            return tuple(embeddings + up(F.silu(down(embeddings)))
                         for down, up in zip(self.down, self.up))


class VisualAdapterController:
    """Instance-local hooks. Native teacher forwards remain completely native."""
    def __init__(self, model, adapter):
        self.model = model
        self.adapter = adapter
        self.mode = 'native'
        self.mask = self.visual_idx = self.text_idx = None
        self.predictions = None
        self.captured = []
        self.checkpoint_layers = False
        self.originals = []
        self.hook = model.model.language_model.register_forward_pre_hook(self._start, with_kwargs=True)
        for index, layer in enumerate(model.model.language_model.layers):
            original = layer.forward
            self.originals.append(original)
            def wrapped(layer, hidden_states, *args, _i=index, _orig=original, **kwargs):
                return self._layer(_i, layer, _orig, hidden_states, *args, **kwargs)
            layer.forward = types.MethodType(wrapped, layer)

    def _start(self, module, args, kwargs):
        h = kwargs.get('inputs_embeds')
        self.predictions = None
        self.visual_idx = self.text_idx = None
        if self.mode == 'native' or h is None or self.mask is None:
            return
        cache = kwargs.get('past_key_values')
        if cache is not None and cache.get_seq_length() > 0:
            return  # Native incremental text decode with hybrid states from prefill.
        assert h.shape[:2] == self.mask.shape and h.shape[0] == 1
        self.visual_idx = self.mask[0].nonzero().flatten()
        self.text_idx = (~self.mask[0]).nonzero().flatten()
        assert self.visual_idx.numel() and self.text_idx.numel()
        if self.mode == 'adapter':
            self.predictions = self.adapter(h.index_select(1, self.visual_idx).detach())
        elif self.mode == 'oracle':
            self.predictions = tuple(self.captured)
            assert len(self.predictions) == len(module.layers)
        elif self.mode == 'capture':
            self.captured = []

    def _layer(self, index, layer, original, hidden, *args, **kwargs):
        if self.mode == 'capture' and self.visual_idx is not None:
            self.captured.append(hidden.index_select(1, self.visual_idx).detach().clone())
        if self.predictions is None:
            return original(hidden, *args, **kwargs)
        assert not args, 'Expected native named decoder arguments'
        visual_idx, text_idx = self.visual_idx, self.text_idx
        prediction = self.predictions[index]

        def run(h, v):
            h = h.index_copy(1, visual_idx, v.to(h.dtype))
            normalized = layer.input_layernorm(h)
            call_kwargs = dict(kwargs)
            positions = call_kwargs.pop('position_embeddings')
            attention_mask = call_kwargs.pop('attention_mask', None)
            position_ids = call_kwargs.pop('position_ids', None)
            cache = call_kwargs.pop('past_key_values', None)
            if layer.block_type == 'linear_attention':
                mixed = layer.linear_attn(normalized, cache_params=cache,
                                          attention_mask=attention_mask, **call_kwargs)
            else:
                mixed, _ = layer.self_attn(normalized, position_embeddings=positions,
                    attention_mask=attention_mask, position_ids=position_ids,
                    past_key_values=cache, **call_kwargs)
            text = h.index_select(1, text_idx) + mixed.index_select(1, text_idx)
            text = text + layer.mlp(layer.post_attention_layernorm(text))
            return h.index_copy(1, text_idx, text)

        if self.checkpoint_layers and torch.is_grad_enabled():
            assert kwargs.get('past_key_values') is None, 'No mutable caches in checkpointed training'
            return checkpoint(run, hidden, prediction, use_reentrant=False)
        return run(hidden, prediction)

    @contextmanager
    def activate(self, mode, mask=None, checkpoint_layers=False):
        assert self.mode == 'native', 'Nested adapter contexts are not supported'
        self.mode, self.mask, self.checkpoint_layers = mode, mask, checkpoint_layers
        try:
            yield self
        finally:
            self.mode, self.mask = 'native', None
            self.predictions = self.visual_idx = self.text_idx = None
            self.checkpoint_layers = False

    def close(self):
        self.hook.remove()
        for layer, original in zip(self.model.model.language_model.layers, self.originals):
            layer.forward = original


# Shared model, image-input and KL utilities for the Qwen3.5 PixMo run.
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch.nn.functional as F



def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def score_evaluation_prediction(prediction, row, metric):
    from src.benchmarks import score_prediction
    score = score_prediction(metric=metric, prediction_text=prediction['prediction_text'],
        answer=row.get('answer'), answers=row.get('answers'), choices=row.get('choices'),
        question=row.get('question'))
    if prediction.get('stopped_by_eos') is False:
        score.update(prediction=None, score=0.0, invalid=True)
    return score


def generate_evaluation_answer(model, processor, inputs, row, spec, config, *, max_new_tokens=None):
    """One pinned generation/scoring path for native, adapter and pruning.

    Store token IDs and actual EOS status. An unfinished response at the length
    cap is invalid; do not extract a convenient letter from its reasoning.
    """
    protocol = config['evaluation_generation']
    cap = int(protocol['max_new_tokens'] if max_new_tokens is None else max_new_tokens)
    assert cap > 0
    assert protocol['do_sample'] is False and protocol['unfinished_response'] == 'invalid_zero'
    output = model.generate(**inputs, do_sample=False, max_new_tokens=cap, use_cache=True,
                            pad_token_id=processor.tokenizer.pad_token_id)
    tokens = output[0, inputs['input_ids'].shape[1]:].tolist()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    finished = bool(tokens and eos is not None and tokens[-1] in eos)
    text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
    score = score_evaluation_prediction(dict(prediction_text=text, stopped_by_eos=finished), row, spec.metric)
    return dict(prediction_text=text, **score, generated_tokens=len(tokens),
                generated_token_ids=tokens, stopped_by_eos=finished,
                hit_generation_limit=not finished and len(tokens) >= cap,
                max_new_tokens=cap)


def load_model(config, device):
    from src.model_setup import load_qwen35
    return load_qwen35(config, device)


def answer_suffix(processor, row):
    eos = processor.tokenizer.eos_token
    answer = str(row['answer']).strip()
    return ' ' + answer + (eos if eos and not answer.endswith(eos) else '')


def prepare_inputs(processor, row, image_root, device, *, question=None, training=False):
    root = Path(row.get('image_root') or image_root)
    paths = row.get('images') or [row['image']]
    assert len(paths) == 1, 'This experiment is single-image only'
    content = [{'type': 'image'} for _ in paths]
    content.append({'type': 'text', 'text': str(row['question'] if question is None else question).strip()})
    prompt = processor.apply_chat_template([{'role': 'user', 'content': content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    images = []
    for path in paths:
        with Image.open(root / path) as im:
            images.append(im.convert('RGB').copy())
    inputs = dict(processor(text=[prompt], images=images, return_tensors='pt'))
    assert 'mm_token_type_ids' in inputs
    prompt_length = inputs['input_ids'].shape[1]
    if training:
        suffix = answer_suffix(processor, row)
        answer_ids = processor.tokenizer(suffix, add_special_tokens=False, return_tensors='pt')['input_ids']
        assert answer_ids.numel() > 0
        inputs['input_ids'] = torch.cat((inputs['input_ids'], answer_ids), dim=1)
        inputs['attention_mask'] = torch.cat((inputs['attention_mask'], torch.ones_like(answer_ids)), dim=1)
        inputs['mm_token_type_ids'] = torch.cat((inputs['mm_token_type_ids'], torch.zeros_like(answer_ids)), dim=1)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}, prompt_length


@torch.no_grad()
def initial_context(model, inputs):
    """One frozen vision pass, exactly shared by teacher and student."""
    embeddings = model.get_input_embeddings()(inputs['input_ids'])
    features = model.model.get_image_features(inputs['pixel_values'], inputs['image_grid_thw'],
                                             return_dict=True).pooler_output
    features = torch.cat(features, dim=0).to(embeddings)
    mask, _ = model.model.get_placeholder_mask(inputs['input_ids'], inputs_embeds=embeddings,
                                               image_features=features)
    embeddings = embeddings.masked_scatter(mask, features)
    positions = model.model.compute_3d_position_ids(input_ids=inputs['input_ids'],
        image_grid_thw=inputs['image_grid_thw'], inputs_embeds=embeddings,
        attention_mask=inputs['attention_mask'], mm_token_type_ids=inputs['mm_token_type_ids'],
        past_key_values=None)
    assert positions is not None
    return dict(inputs_embeds=embeddings, position_ids=positions,
                attention_mask=inputs['attention_mask'], use_cache=False)


@torch.no_grad()
def teacher_targets(model, context, prompt_length, targets, topk=1024, temperature=2.):
    hidden = model.model.language_model(**context).last_hidden_state[:, prompt_length-1:-1]
    indices, probabilities = [], []
    for start in range(0, hidden.shape[1], 32):
        logits = model.lm_head(hidden[:, start:start+32]).float()
        idx = logits.topk(topk, dim=-1).indices
        gold = targets[:, start:start+32, None]
        contains = (idx == gold).any(-1, keepdim=True)
        with_gold = torch.cat((idx[..., :-1], gold), dim=-1)
        idx = torch.where(contains, idx, with_gold)
        indices.append(idx)
        probabilities.append((logits.gather(-1, idx) / temperature).softmax(-1))
    return torch.cat(indices, 1), torch.cat(probabilities, 1)


def student_loss(model, context, prompt_length, indices, probabilities, temperature=2.):
    hidden = model.model.language_model(**context).last_hidden_state[:, prompt_length-1:-1]
    loss = hidden.new_zeros((), dtype=torch.float32)
    def chunk(h, idx, probability):
        logits = model.lm_head(h).float().gather(-1, idx) / temperature
        return F.kl_div(logits.log_softmax(-1), probability, reduction='sum') * temperature**2
    for start in range(0, hidden.shape[1], 32):
        loss = loss + checkpoint(chunk, hidden[:, start:start+32], indices[:, start:start+32],
                                probabilities[:, start:start+32], use_reentrant=False)
    return loss / hidden.shape[1]
