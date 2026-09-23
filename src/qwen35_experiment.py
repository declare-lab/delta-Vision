"""Shared model, image-input and KL utilities for the Qwen3.5 PixMo run."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.qwen35_embedding import StaticVisualAdapter, VisualAdapterController, install_fast_kernels


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
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    kernels = install_fast_kernels()
    processor = AutoProcessor.from_pretrained(config['model_path'], local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(config['model_path'],
        dtype=torch.bfloat16, attn_implementation='flash_attention_2',
        device_map={'': str(device)}, local_files_only=True).eval().requires_grad_(False)
    from src.qwen_deepstack import disable_qwen_deepstack
    disable_qwen_deepstack(model)
    c = model.config.text_config
    assert c.hidden_size == 2560 and c.num_hidden_layers == 32
    assert c.layer_types.count('linear_attention') == 24 and c.layer_types.count('full_attention') == 8
    assert not model.config.vision_config.deepstack_visual_indexes
    assert model.model.language_model.config._attn_implementation == 'flash_attention_2'
    assert model.model.visual.config._attn_implementation == 'flash_attention_2'
    adapter = StaticVisualAdapter(rank=config['rank']).to(device)
    assert sum(p.numel() for p in adapter.parameters()) == 20971520
    controller = VisualAdapterController(model, adapter)
    print('QWEN35_BACKBONE_FROZEN', json.dumps(kernels), flush=True)
    return processor, model, adapter, controller


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
