from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import SinkCache


class PartialLlamaSinkCache(SinkCache):
    """
    Distributed implementation of sink cache (supports multiple different queries running at the same time)
    """
    def __init__(self, window_length: int, num_sink_tokens: int):
        super.__init__(window_length, num_sink_tokens)

        self.key_cache: Dict[str, List[torch.Tensor]] = {}
        self.value_cache: Dict[str, List[torch.Tensor]] = {}
        self._cos_cache: Dict[str, torch.Tensor] = {}
        self._sin_cache: Dict[str, torch.Tensor] = {}
        self._seen_tokens: Dict[str, int] = {}
        self.cos_sin_rerotation_cache: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}

    def _get_rerotation_cos_sin(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, generation_id: str | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if generation_id is None:
            raise ValueError("generation_id not provided")

        cos_sin_rerotation_cache = self.cos_sin_rerotation_cache.get(generation_id)

        if cos_sin_rerotation_cache is None:
            raise ValueError("cos_sin_rerotation_cache not found")

        if key_states.shape[-2] not in cos_sin_rerotation_cache:
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

            original_cos = cos[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_cos = cos[self.num_sink_tokens : -key_states.shape[-2]]
            original_sin = sin[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_sin = sin[self.num_sink_tokens : -key_states.shape[-2]]
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            cos_sin_rerotation_cache[key_states.shape[-2]] = (
                rerotation_cos.to(key_states.dtype).unsqueeze(0),
                rerotation_sin.to(key_states.dtype).unsqueeze(0),
            )
        
        return cos_sin_rerotation_cache[key_states.shape[-2]]
    
    def get_seq_length(self, layer_idx: int | None = 0, generation_id: str | None = None) -> int:
        if not generation_id:
            raise ValueError("generation_id not provided")
        
        key_cache = self.key_cache.get(generation_id)
        
        if not key_cache:
            return 0
        
        if len(key_cache) <= layer_idx:
            return 0
        
        return key_cache[layer_idx].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache_kwargs is None or "generation_id" not in cache_kwargs:
            raise ValueError("generation_id not found in cache_kwargs")
        
        generation_id = cache_kwargs.get("generation_id")
        sin = cache_kwargs.get("sin")
        cos = cache_kwargs.get("cos")

        if generation_id not in self.key_cache:
            self.key_cache[generation_id] = []
            self.value_cache[generation_id] = []
            self._cos_cache[generation_id] = None
            self._sin_cache[generation_id] = None
            self._seen_tokens[generation_id] = 0
            self.cos_sin_rerotation_cache[generation_id] = {}

        if layer_idx == 0:
            self._seen_tokens[generation_id] += key_states.shape[-2]

        using_rope = sin is not None and cos is not None

        if using_rope and layer_idx == 0:
            if cos.dim() == 2:
                self._cos_cache[generation_id] = cos
                self._sin_cache[generation_id] = sin
            else:
                if self._cos_cache[generation_id] is None:
                    self._cos_cache[generation_id] = cos[0, ...]
                    self._sin_cache[generation_id] = sin[0, ...]
                elif self._cos_cache[generation_id].shape[0] < self.window_length:
                    self._cos_cache[generation_id] = torch.cat([self._cos_cache[generation_id], cos[0, ...]], dim=0)
                    self._sin_cache[generation_id] = torch.cat([self._sin_cache[generation_id], sin[0, ...]], dim=0)

        if len(self.key_cache[generation_id]) <= layer_idx:
            self.key_cache[generation_id].append(key_states)
            self.value_cache[generation_id].append(value_states)
        
        elif key_states.shape[-2] + self.get_seq_length(layer_idx, generation_id) < self.window_length:
            self.key_cache[generation_id][layer_idx] = torch.cat([self.key_cache[generation_id][layer_idx], key_states], dim=-2)
            self.value_cache[generation_id][layer_idx] = torch.cat([self.value_cache[generation_id][layer_idx], value_states], dim=-2)
        
        else:
            keys_to_keep = self.key_cache[generation_id][layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + key_states.shape[-2] :
            ]

            if using_rope:
                rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(
                    key_states, 
                    self._cos_cache[generation_id][: self.window_length], 
                    self._sin_cache[generation_id][: self.window_length], 
                    generation_id
                )

                keys_to_keep = self._apply_key_rotary_pos_emb(keys_to_keep, rerotation_cos, rerotation_sin)
                
            sink_keys = self.key_cache[generation_id][layer_idx][:, :, : self.num_sink_tokens]
            self.key_cache[generation_id][layer_idx] = torch.cat([sink_keys, keys_to_keep, key_states], dim=-2)

            sink_values = self.value_cache[generation_id][layer_idx][:, :, : self.num_sink_tokens]
            values_to_keep = self.value_cache[generation_id][layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + value_states.shape[-2] :
            ]
            self.value_cache[generation_id][layer_idx] = torch.cat([sink_values, values_to_keep, value_states], dim=-2)
        
        return self.key_cache[generation_id][layer_idx], self.value_cache[generation_id][layer_idx]
