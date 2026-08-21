from __future__ import annotations

import types

import torch

from .lcd import select_visual_tokens


def _language_model(model):
    """Resolve the text stack from either bare Transformers or a PEFT wrapper."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.language_model


def install_lcd_forward(model) -> None:
    """Patch the Qwen3-VL text stack with layer-intermediate visual pruning."""
    language_model = _language_model(model)
    if getattr(language_model, "_epic_installed", False):
        return
    language_model._epic_state = None
    language_model._epic_keep_indices = None

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, cache_position=None, visual_pos_masks=None,
                deepstack_visual_embeds=None, **kwargs):
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.models.qwen3_vl.modeling_qwen3_vl import create_causal_mask

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        text_position_ids = position_ids[0] if position_ids.shape[0] == 3 else position_ids[0]
        if position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        raw_mask = attention_mask
        hidden_states = inputs_embeds
        state = self._epic_state
        self._epic_keep_indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)

        def make_mask():
            return create_causal_mask(config=self.config, input_embeds=hidden_states,
                                      attention_mask=raw_mask, cache_position=cache_position,
                                      past_key_values=past_key_values,
                                      position_ids=text_position_ids)

        causal_mask = make_mask()
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_idx, decoder_layer in enumerate(self.layers):
            # Prune only multimodal prefill passes. Decode passes have no
            # visual_pos_masks and must reuse the already-compressed context.
            if (state is not None and visual_pos_masks is not None
                    and layer_idx == state["layer"] and hidden_states.shape[1] > 1):
                keep = select_visual_tokens(hidden_states, visual_pos_masks, state["ratio"],
                                            state["method"], state["min_keep"])
                self._epic_keep_indices = self._epic_keep_indices[keep]
                hidden_states = hidden_states[:, keep]
                position_ids = position_ids[:, :, keep]
                text_position_ids = text_position_ids[:, keep]
                raw_mask = raw_mask[:, keep] if raw_mask is not None and raw_mask.ndim == 2 else None
                visual_pos_masks = visual_pos_masks[:, keep]
                cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
                causal_mask = make_mask()
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
            hidden_states = decoder_layer(
                hidden_states, attention_mask=causal_mask, position_ids=text_position_ids,
                past_key_values=past_key_values, cache_position=cache_position,
                position_embeddings=position_embeddings, **kwargs,
            )
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                embeds = deepstack_visual_embeds[layer_idx]
                if embeds.shape[0] != int(visual_pos_masks.sum()):
                    original_visual = self._epic_keep_indices[visual_pos_masks[0]].new_tensor(
                        self._epic_keep_indices[visual_pos_masks[0]]
                    )
                    # Map retained original sequence positions to original visual ordinals.
                    full_visual = state["full_visual_mask"][0].nonzero().flatten()
                    ordinals = torch.searchsorted(full_visual, original_visual)
                    embeds = embeds[ordinals]
                hidden_states = self._deepstack_process(hidden_states, visual_pos_masks, embeds)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    language_model.forward = types.MethodType(forward, language_model)
    language_model._epic_installed = True


def set_pruning(model, *, layer: int, ratio: float, method: str, min_keep: int,
                full_visual_mask: torch.Tensor) -> None:
    _language_model(model)._epic_state = {
        "layer": int(layer), "ratio": float(ratio), "method": method,
        "min_keep": int(min_keep), "full_visual_mask": full_visual_mask,
    }


def clear_pruning(model) -> None:
    _language_model(model)._epic_state = None


def get_keep_indices(model) -> torch.Tensor:
    return _language_model(model)._epic_keep_indices
