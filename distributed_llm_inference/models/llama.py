import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      LlamaConfig,
                                                      LlamaDecoderLayer,
                                                      LlamaMLP, LlamaRMSNorm,
                                                      repeat_kv, rotate_half)

from distributed_llm_inference.utils.cuda import \
    make_inference_graphed_callable


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class OptimizedLlamaInferenceAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rotary_graph = None
    
    def _optimized_apply_rotary(self, query_states, key_states, cos, sin):
        if self._rotary_graph is None:
            self._rotary_graph = make_inference_graphed_callable(
                apply_rotary_pos_emb, sample_args=(query_states, key_states, cos, sin)
            )
        
        return self._rotary_graph(query_states, key_states, cos, sin)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        
        bsz, q_len, _ = hidden_states.size()

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            past_seen_tokens = past_key_value[0].shape[2] if past_key_value is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)
            
        cos, sin = self.rotary_emb(value_states, position_ids)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        if q_len == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
            query_states, key_states = self._optimized_apply_rotary(query_states, key_states, cos, sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states)
        
        key_states = repeat_kv(key_states, self.num_heads)
        value_states = repeat_kv(value_states, self.num_heads)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value
    

class OptimizedLlamaInferenceDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        torch.nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = OptimizedLlamaInferenceAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(self.hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(self.hidden_size)

        self.pre_attn_graph = None
        self.post_attn_graph = None
    
    def _optimized_input_norm(self, hidden_states):
        if self.pre_attn_graph is None:
            self.pre_attn_graph = make_inference_graphed_callable(
                self.input_layernorm.forward, sample_args=(hidden_states,)
            )
        
        return self.pre_attn_graph(hidden_states)

    def _optimized_output_norm(self, hidden_states, attn_output):
        if self.post_attn_graph is None:
            self.post_attn_graph = make_inference_graphed_callable(
                self.post_attention_layernorm.forward, sample_args=(hidden_states, attn_output)
            )
        
        return self.post_attn_graph(hidden_states, attn_output)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        _, q_len, _ = hidden_states.size()

        if q_len == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
            hidden_states = self._optimized_input_norm(hidden_states)
        else:
            hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, _, key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs
        )

        hidden_states = hidden_states + residual
        residual = hidden_states

        if q_len == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
            hidden_states = self._optimized_output_norm(hidden_states, residual)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states + residual)
        
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return (hidden_states, key_value)


class LlamaBlock(OptimizedLlamaInferenceDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        batch_size, seq_length, _ = hidden_states.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        past_key_value = layer_past
        if past_key_value is not None:
            past_key_values_length = past_key_value[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            past_key_value = self._reorder_cache_from_bloom_to_llama(past_key_value, batch_size, past_key_values_length)

        assert position_ids is None

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=hidden_states,
            past_key_values_length=past_key_values_length,
        )

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )

        return outputs