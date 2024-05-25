# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Gemma model."""
import os
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.models.gemma.configuration_gemma import GemmaConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

import sys

home_dir = os.getenv("HOME_DIR")
sys.path.append(home_dir)
from passes import _pass_fn


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GemmaConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class MultiLinear(nn.Module):
    def __init__(self, channels, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels, in_features, out_features))

    def forward(self, x):
        return x @ self.weight


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * (1 + self.weight)


ALL_LAYERNORM_LAYERS.append(GemmaRMSNorm)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base
                ** (
                    torch.arange(
                        0, self.dim, 2, dtype=torch.int64, device=x.device
                    ).float()
                    / self.dim
                )
            )
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=None, think_tokens=None, unsqueeze_dim=1
):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if think_tokens is not None:
        k_prefix = k[..., :think_tokens, :]
        k = k[..., think_tokens:, :]
        q_prefix = q[..., :think_tokens, :]
        q = q[..., think_tokens:, :]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    if think_tokens is not None:
        q_embed = torch.cat((q_prefix, q_embed), dim=-2)
        k_embed = torch.cat((k_prefix, k_embed), dim=-2)
    return q_embed, k_embed


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Gemma
class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Ignore copy
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prompt_lengths: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # if cache_position is not None and cache_position[0] == 0:
        #     # add range(position_ids[-1], position_ids[-1] + self.config.think_tokens) to position_ids
        #     position_ids = torch.cat(
        #         [
        #             position_ids,
        #             torch.arange(
        #                 position_ids[0, -1] + 1,
        #                 position_ids[0, -1] + 1 + self.config.think_tokens,
        #                 device=position_ids.device,
        #             ).unsqueeze(0),
        #         ],
        #         dim=-1,
        #     )

        if self.layer_idx == 0:
            # print(value_states.shape, position_ids.shape)
            # what was that??
            pass

        if self.config.think_pos == 1 and (
            cache_position is None or cache_position[0] == 0
        ):
            assert self.config.think_type == 2
            cos, sin = self.rotary_emb(
                value_states[:, :, self.config.think_tokens :],
                position_ids[:, : -self.config.think_tokens],
                seq_len=None,
            )
        else:
            cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            None,
            (
                self.config.think_tokens
                if (
                    self.config.think_pos == 1
                    and (cache_position is None or cache_position[0] == 0)
                )
                else None
            ),
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            if cache_position is not None and cache_position[0] > 0:
                causal_mask = attention_mask[
                    :, :, cache_position, : key_states.shape[-2]
                ]
            else:
                causal_mask = attention_mask
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


GEMMA_ATTENTION_CLASSES = {
    "eager": GemmaAttention,
}


# Copied from transformers.models.llama.modeling_llama.LlamaDecoderLayer with LLAMA->GEMMA,Llama->Gemma
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config

        self.self_attn = GEMMA_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prompt_lengths: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            prompt_lengths=prompt_lengths,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


GEMMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GemmaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Gemma Model outputting raw hidden-states without any specific head on top.",
    GEMMA_START_DOCSTRING,
)
class GemmaPreTrainedModel(PreTrainedModel):
    config_class = GemmaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keep_in_fp32_modules = ["inv_freq", "rotary_emb", "cos_cached", "sin_cached"]
    _no_split_modules = ["GemmaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _setup_cache(
        self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None
    ):
        if (
            self.config._attn_implementation == "flash_attention_2"
            and cache_cls == StaticCache
        ):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        if (
            max_cache_len > self.model.causal_mask.shape[-1]
            or self.device != self.model.causal_mask.device
        ):
            causal_mask = torch.full(
                (max_cache_len, max_cache_len), fill_value=1, device=self.device
            )
            self.register_buffer(
                "causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False
            )

        for layer in self.model.layers:
            weights = layer.self_attn.o_proj.weight
            layer.self_attn.past_key_value = cache_cls(
                self.config,
                max_batch_size,
                max_cache_len,
                device=weights.device,
                dtype=weights.dtype,
            )

    def _reset_cache(self):
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None


GEMMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Gemma Model outputting raw hidden-states without any specific head on top.",
    GEMMA_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaModel with LLAMA->GEMMA,Llama->Gemma
class GemmaModel(GemmaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GemmaDecoderLayer`]

    Args:
        config: GemmaConfig
    """

    def __init__(self, config: GemmaConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        if not hasattr(config, "proj_type"):
            config.proj_type = 0
        self.proj_type = config.proj_type
        if not hasattr(config, "think_tokens"):
            config.think_tokens = 1
        self.think_tokens = config.think_tokens
        if not hasattr(config, "proj_channels"):
            config.proj_channels = 16
        if not hasattr(config, "skip_bidir"):
            config.skip_bidir = 0
        self.proj_channels = config.proj_channels
        if self.think_tokens > 0:
            if self.proj_type == 0:
                self.think_embeds = nn.Embedding(
                    config.think_tokens, config.hidden_size
                )
            elif self.proj_type == 4:
                self.think_embeds = nn.Embedding(
                    config.think_tokens * config.proj_channels, config.hidden_size
                )
                self.think_embeds_proj = nn.Linear(
                    config.hidden_size,
                    config.proj_channels,
                    bias=hasattr(config, "proj_bias") and config.proj_bias == 1,
                )
            elif self.proj_type == 1 or self.proj_type == 3:
                if self.think_tokens > 1 and self.proj_type == 1:
                    std = self.config.initializer_range
                    self.think_embeds = MultiLinear(
                        self.think_tokens, config.hidden_size, config.hidden_size
                    )
                    self.think_embeds.weight.data.normal_(mean=0.0, std=std)
                else:
                    self.think_embeds = nn.Linear(
                        config.hidden_size, config.hidden_size, bias=False
                    )
            elif self.proj_type == 2:
                if self.think_tokens > 1:
                    raise NotImplementedError()
                else:
                    self.think_embeds = GemmaMLP(config)
            else:
                raise ValueError(f"Invalid `proj_type` {self.proj_type}")
        if not hasattr(config, "think_type"):
            config.think_type = 0
        if hasattr(self.config, "n_pass") and self.config.n_pass > 0:
            self.prepass_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.prepass_proj.weight.data = torch.eye(self.config.hidden_size)

        # Register a causal mask to separate causal and padding mask creation. Merging happens in the attention class.
        # NOTE: This is not friendly with TorchScript, ONNX, ExportedProgram serialization for very large `max_position_embeddings`.
        causal_mask = torch.full(
            (config.max_position_embeddings, config.max_position_embeddings),
            fill_value=True,
            dtype=torch.bool,
        )
        self.register_buffer(
            "causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False
        )
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(GEMMA_INPUTS_DOCSTRING)
    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor,
        prompt_lengths: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        enforce_bidir: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # add think tokens
        if self.think_tokens > 0 and (cache_position is None or cache_position[0] == 0):

            if self.proj_type == 0:
                _think_embeds = self.think_embeds(
                    torch.tensor(range(self.think_tokens), device=inputs_embeds.device)
                )
            elif self.proj_type >= 1:
                assert inputs_embeds.shape[0] == 1

                if self.config.attn_type == 0:
                    _causal_mask = self._update_causal_mask(None, inputs_embeds)[
                        ..., : prompt_lengths[0], : prompt_lengths[0]
                    ]
                elif self.config.attn_type == 2:
                    _causal_mask = None
                else:
                    raise ValueError(f"Invalid `attn_type` {self.config.attn_type}")

                _position_ids = torch.arange(
                    0,
                    prompt_lengths[0],
                    device=inputs_embeds.device,
                ).unsqueeze(0)
                _inputs_embeds = inputs_embeds[:, : prompt_lengths[0]]

                proj_idx = (
                    0 if not hasattr(self.config, "proj_idx") else self.config.proj_idx
                )
                for decoder_layer in self.layers[: len(self.layers) - proj_idx]:
                    if self.gradient_checkpointing and self.training:
                        layer_outputs = self._gradient_checkpointing_func(
                            decoder_layer.__call__,
                            _inputs_embeds,
                            prompt_lengths,
                            _causal_mask,
                            _position_ids,
                        )
                    else:
                        layer_outputs = decoder_layer(
                            _inputs_embeds,
                            prompt_lengths,
                            attention_mask=_causal_mask,
                            position_ids=_position_ids,
                        )
                    hidden_states = layer_outputs[0]
                hidden_states = self.norm(hidden_states)

                if self.proj_type == 3:
                    _think_embeds = self.think_embeds(
                        hidden_states[
                            0,
                            prompt_lengths[0] - self.think_tokens : prompt_lengths[0],
                            :,
                        ]
                    )
                elif self.proj_type == 4:
                    _projected = self.think_embeds_proj(hidden_states)
                    _projected = nn.functional.softmax(
                        _projected, dim=-1, dtype=torch.float32
                    ).to(_projected.dtype)
                    _think_embeds = self.think_embeds(
                        torch.tensor(
                            range(self.think_tokens * self.proj_channels),
                            device=inputs_embeds.device,
                        )
                    ).view(
                        self.think_tokens, self.proj_channels, self.config.hidden_size
                    )
                    _think_embeds = torch.matmul(_projected, _think_embeds).sum(dim=-2)
                else:
                    if self.think_tokens > 1:
                        _think_embeds = (
                            self.think_embeds(
                                hidden_states[:1, prompt_lengths[0] - 1, :]
                            )
                        ).squeeze(1)
                    else:
                        _think_embeds = self.think_embeds(
                            hidden_states[:1, prompt_lengths[0] - 1, :]
                        )

            if self.config.think_type == 2:
                prefix = _think_embeds.unsqueeze(0).expand(
                    inputs_embeds.shape[0], self.think_tokens, self.config.hidden_size
                )
                inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)
            else:
                new_batch = []
                for i in range(inputs_embeds.shape[0]):
                    if self.config.think_type < 2:
                        shift = 0 if self.config.think_type == 1 else 1
                        _tokens = [
                            inputs_embeds[i, : prompt_lengths[i] - shift],
                            _think_embeds,
                            inputs_embeds[i, prompt_lengths[i] - shift :],
                        ]
                    elif self.config.think_type > 2:
                        shift = 0 if self.config.think_type == 4 else 1
                        _pre_count = _think_embeds.shape[0] // 2
                        _pre = _think_embeds[:_pre_count]
                        _post = _think_embeds[_pre_count:]
                        if self.config.think_type == 5:
                            _tokens = [
                                inputs_embeds[i, :1],
                                _pre,
                                inputs_embeds[i, 1 : prompt_lengths[i] - shift],
                                _post,
                                inputs_embeds[i, prompt_lengths[i] - shift :],
                            ]
                        else:
                            _tokens = [
                                _pre,
                                inputs_embeds[i, : prompt_lengths[i] - shift],
                                _post,
                                inputs_embeds[i, prompt_lengths[i] - shift :],
                            ]
                    new_batch.append(torch.cat(_tokens))
                inputs_embeds = torch.stack(new_batch, dim=0)

        seq_len = inputs_embeds.shape[1]

        past_seen_tokens = 0
        if past_key_values is not None:
            past_seen_tokens = past_key_values[0][0].shape[2]

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_len,
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # update think token attn mask
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)
        if cache_position is None or cache_position[0] == 0:
            new_causal_mask = []
            for i in range(inputs_embeds.shape[0]):
                _mask = causal_mask[0, 0, :seq_len, :seq_len]
                if self.config.attn_type > 0:
                    if self.config.attn_type == 1:
                        assert self.config.think_type >= 2
                        index_start = 0
                        index_end = self.think_tokens
                    elif self.config.attn_type == 2:
                        index_start = 0
                        index_end = self.think_tokens + prompt_lengths[i]
                    _mask[
                        index_start:index_end, : self.think_tokens + prompt_lengths[i]
                    ] = 0.0
                new_causal_mask.append(_mask)
            causal_mask = torch.stack(new_causal_mask, dim=0).unsqueeze(1)

        # embed positions
        hidden_states = inputs_embeds

        # normalized
        hidden_states = hidden_states * (self.config.hidden_size**0.5)

        # pre passes
        if hasattr(self.config, "n_pass") and self.config.n_pass > 0:
            assert hidden_states.shape[0] == 1

            _inputs_embeds = hidden_states[..., : prompt_lengths[0], :]
            _causal_mask = causal_mask[..., : prompt_lengths[0], : prompt_lengths[0]]
            _position_ids = position_ids[..., : prompt_lengths[0]]

            for decoder_layer in self.layers:  # [-4:-2]:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        _inputs_embeds,
                        prompt_lengths,
                        _causal_mask,
                        _position_ids,
                    )
                else:
                    layer_outputs = decoder_layer(
                        _inputs_embeds,
                        prompt_lengths,
                        attention_mask=_causal_mask,
                        position_ids=_position_ids,
                    )
                _inputs_embeds = layer_outputs[0]
            _inputs_embeds = self.prepass_proj(_inputs_embeds)
            hidden_states = torch.cat(
                [_inputs_embeds, hidden_states[:, prompt_lengths[0] :]], dim=1
            )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    prompt_lengths,
                    (
                        (causal_mask.transpose(-1, -2) if self.config.ablation == 420 else torch.zeros_like(causal_mask))
                        if enforce_bidir
                        and (
                            self.config.skip_bidir >= 0
                            and idx >= self.config.skip_bidir
                            or self.config.skip_bidir < 0
                            and idx < -self.config.skip_bidir
                        )
                        else causal_mask
                    ),
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    prompt_lengths,
                    attention_mask=(
                        (causal_mask.transpose(-1, -2) if self.config.ablation == 420 else torch.zeros_like(causal_mask))
                        if enforce_bidir
                        and (
                            self.config.skip_bidir >= 0
                            and idx >= self.config.skip_bidir
                            or self.config.skip_bidir < 0
                            and idx < -self.config.skip_bidir
                        )
                        else causal_mask
                    ),
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # remove think tokens
        if self.think_tokens > 0 and (cache_position is None or cache_position[0] == 0):
            if self.config.think_type == 2:
                hidden_states = hidden_states[:, self.think_tokens :]
            else:
                new_hidden_states = []
                for i in range(hidden_states.shape[0]):
                    if self.config.think_type < 2:
                        _tokens = [
                            hidden_states[i, : prompt_lengths[i] - 1],
                            hidden_states[
                                i, prompt_lengths[i] - 1 + self.think_tokens :
                            ],
                        ]
                    elif self.config.think_type > 2:
                        _pre_count = _think_embeds.shape[0] // 2
                        if self.config.think_type == 5:
                            _tokens = [
                                hidden_states[i, :1],
                                hidden_states[
                                    i, _pre_count + 1 : prompt_lengths[i] - 1
                                ],
                                hidden_states[
                                    i,
                                    prompt_lengths[i]
                                    - 1
                                    + self.think_tokens
                                    - _pre_count :,
                                ],
                            ]
                        else:
                            _tokens = [
                                hidden_states[i, _pre_count : prompt_lengths[i] - 1],
                                hidden_states[
                                    i,
                                    prompt_lengths[i]
                                    - 1
                                    + self.think_tokens
                                    - _pre_count :,
                                ],
                            ]
                    new_hidden_states.append(torch.cat(_tokens))
                hidden_states = torch.stack(new_hidden_states, dim=0)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype
        device = input_tensor.device

        # support going beyond cached `max_position_embedding`
        if seq_length > self.causal_mask.shape[-1]:
            causal_mask = torch.full(
                (2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]),
                fill_value=1,
            )
            self.register_buffer(
                "causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False
            )

        # We use the current dtype to avoid any overflows
        min_dtype = torch.finfo(dtype).min
        causal_mask = (
            self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype)
            * min_dtype
        )

        causal_mask = causal_mask.to(dtype=dtype, device=device)
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[
                :, None, None, :
            ].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        if self.config._attn_implementation == "sdpa" and attention_mask is not None:
            # TODO: For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
            is_tracing = (
                torch.jit.is_tracing()
                or isinstance(input_tensor, torch.fx.Proxy)
                or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                # Attend to all tokens in masked rows from the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = causal_mask.mul(
                    ~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True)
                ).to(dtype)

        return causal_mask


class PassScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.threshold = 0.01 if config.pass_type in [4, 7, 8, 9, 104] else 1.0
        self._ratio = nn.Parameter(torch.zeros(config.num_hidden_layers))
        self._denominator = nn.Parameter(torch.zeros(config.num_hidden_layers))

        # it's such a mess
        is_newlinear = (config.pass_type >= 600 and config.pass_type < 700) or (config.pass_type >= 800 and config.pass_type < 900)

        if config.pass_type == 300:
            self.threshold = torch.finfo(torch.bfloat16).eps
            self.weight = nn.Parameter(torch.zeros(config.num_hidden_layers))
            self.weight_og = nn.Parameter(torch.zeros(config.num_hidden_layers))
            self.weight.data.fill_(self.threshold)
            self.weight_og.data.fill_(self.threshold)
        elif (
            config.pass_type in [5, 9, 10]
            or is_newlinear
        ):
            if (
                is_newlinear
                and (self.config.pass_type % 10) % 4 >= 2 or (config.pass_type >= 800 and config.pass_type < 900)
            ):
                if not hasattr(config, 's_init'):
                    config.s_init = 0.01
                self.threshold = config.s_init
            self.weight = nn.Parameter(torch.zeros(config.num_hidden_layers))
            if config.pass_type >= 610 and config.pass_type < 620:
                self.weight = nn.Parameter(
                    torch.zeros(
                        config.num_hidden_layers, config.num_key_value_heads, 1, 1
                    )
                )
                self._ratio = nn.Parameter(
                    torch.zeros(config.num_hidden_layers, config.num_key_value_heads)
                )
                self._denominator = nn.Parameter(
                    torch.zeros(config.num_hidden_layers, config.num_key_value_heads)
                )
            if config.pass_type >= 620 and config.pass_type < 630:
                self.weight = nn.Parameter(torch.zeros(1))
                self._ratio = nn.Parameter(torch.zeros(1))
                self._denominator = nn.Parameter(torch.zeros(1))
            self.weight.data.fill_(self.threshold)
        elif config.pass_type == 6 or config.pass_type == 7:
            self.weight = nn.Parameter(
                torch.zeros(
                    config.num_hidden_layers,
                    config.num_key_value_heads,
                    1,
                    config.head_dim,
                )
            )
        elif (
            config.pass_type == 8 or config.pass_type >= 520 and config.pass_type < 600
        ):
            self.weight = nn.Parameter(
                torch.zeros(
                    config.num_hidden_layers,
                    config.num_key_value_heads,
                    1,
                    1,
                )
            )
            self._ratio = nn.Parameter(
                torch.zeros(config.num_hidden_layers, config.num_key_value_heads)
            )
            self._denominator = nn.Parameter(
                torch.zeros(config.num_hidden_layers, config.num_key_value_heads)
            )
            self.weight.data.fill_(0.0001)
        else:
            self.weight = nn.Parameter(torch.zeros(config.num_hidden_layers))
            self.weight.data.fill_(0.0001)

        print(f"Threshold: {self.threshold}")

    def forward(self, og_hidden_states, hidden_states, idx):
        if self.config.pass_type >= 400 and self.weight.dtype != torch.float32:
            self.weight.data = self.weight.data.to(torch.float32)

        if self.config.pass_type >= 100 and self.config.pass_type < 200:
            return (
                og_hidden_states * self.threshold
                + hidden_states * self.weight[idx] * self.weight[idx]
            ) / (self.threshold + self.weight[idx] * self.weight[idx])
        elif self.config.pass_type >= 200 and self.config.pass_type < 300:
            return (
                og_hidden_states * (1.0 - self.weight[idx])
                + hidden_states * self.weight[idx]
            )
        elif self.config.pass_type >= 300 and self.config.pass_type < 400:
            return (
                og_hidden_states * self.weight_og[idx] * self.weight_og[idx]
                + hidden_states * self.weight[idx] * self.weight[idx]
            ) / (
                self.weight_og[idx] * self.weight_og[idx]
                + self.weight[idx] * self.weight[idx]
            )
        elif self.config.pass_type >= 400 and self.config.pass_type < 500:
            if self.config.pass_type >= 404:
                self.threshold = 0.01
            dtype = og_hidden_states.dtype
            temp_dtype = torch.float32
            scale = 1.0 + self.weight[idx].to(dtype=temp_dtype)
            scale = scale * scale
            if (
                self.config.pass_type == 401
                or self.config.pass_type == 403
                or self.config.pass_type >= 404
            ):
                scale = scale - 1.0
            og_hidden_states = og_hidden_states.to(dtype=temp_dtype)
            hidden_states = hidden_states.to(dtype=temp_dtype)

            self._ratio.requires_grad = False
            self._ratio.data = self._ratio.data.detach()
            if (
                self.config.pass_type == 402
                or self.config.pass_type == 403
                or self.config.pass_type == 405
            ):
                result = (og_hidden_states * self.threshold + hidden_states * scale) / (
                    self.threshold + scale.abs()
                )
                self._ratio.data[idx] = scale / (self.threshold + scale.abs())
            else:
                result = (og_hidden_states * scale + hidden_states * self.threshold) / (
                    self.threshold + scale.abs()
                )
                self._ratio.data[idx] = self.threshold / (self.threshold + scale.abs())
            self._denominator.requires_grad = False
            self._denominator.data = self._denominator.data.detach()
            self._denominator.data[idx] = (
                self.threshold + scale.abs()
            ) / self.threshold
            return result.to(dtype=dtype)
        elif self.config.pass_type >= 500 and self.config.pass_type < 600:
            if self.config.pass_type % 10 >= 2:
                self.threshold = 0.01
            dtype = og_hidden_states.dtype
            temp_dtype = torch.float32
            scale = self.weight[idx].to(dtype=temp_dtype)
            og_hidden_states = og_hidden_states.to(dtype=temp_dtype)
            hidden_states = hidden_states.to(dtype=temp_dtype)

            self._ratio.requires_grad = False
            self._ratio.data = self._ratio.data.detach()
            _scale = scale.abs() if self.config.pass_type >= 510 else scale
            if self.config.pass_type % 10 == 1 or self.config.pass_type % 10 == 3:
                result = (og_hidden_states * self.threshold + hidden_states * scale) / (
                    self.threshold + _scale
                )
                self._ratio.data[idx] = (scale / (self.threshold + _scale)).squeeze()
            else:
                result = (og_hidden_states * scale + hidden_states * self.threshold) / (
                    self.threshold + _scale
                )
                self._ratio.data[idx] = (
                    self.threshold / (self.threshold + _scale)
                ).squeeze()
            self._denominator.requires_grad = False
            self._denominator.data = self._denominator.data.detach()
            self._denominator.data[idx] = (
                (self.threshold + _scale) / self.threshold
            ).squeeze()
            return result.to(dtype=dtype)
        elif self.config.pass_type >= 600 and self.config.pass_type < 700:
            if self.config.pass_type >= 620:
                idx = 0
            dtype = og_hidden_states.dtype
            temp_dtype = torch.float32
            scale = self.weight[idx].to(dtype=temp_dtype)
            og_hidden_states = og_hidden_states.to(dtype=temp_dtype)
            hidden_states = hidden_states.to(dtype=temp_dtype)

            self._ratio.requires_grad = False
            self._ratio.data = self._ratio.data.detach()
            _scale = scale.abs()
            if self.config.pass_type % 10 >= 4:
                scale = scale.abs()
            if self.config.pass_type % 2 == 1:
                result = (og_hidden_states * self.threshold + hidden_states * scale) / (
                    self.threshold + _scale
                )
                self._ratio.data[idx] = (scale / (self.threshold + _scale)).squeeze()
            else:
                result = (og_hidden_states * scale + hidden_states * self.threshold) / (
                    self.threshold + _scale
                )
                self._ratio.data[idx] = (
                    self.threshold / (self.threshold + _scale)
                ).squeeze()
            self._denominator.requires_grad = False
            self._denominator.data = self._denominator.data.detach()
            self._denominator.data[idx] = (
                (self.threshold + _scale) / self.threshold
            ).squeeze()
            return result.to(dtype=dtype)
        elif self.config.pass_type >= 700 and self.config.pass_type < 800:
            dtype = og_hidden_states.dtype
            temp_dtype = torch.float32
            scale = self.weight[idx].to(dtype=temp_dtype)
            og_hidden_states = og_hidden_states.to(dtype=temp_dtype)
            hidden_states = hidden_states.to(dtype=temp_dtype)

            meta_scale = 1.0
            if self.config.pass_type == 701:
                meta_scale = 50.0
            if self.config.pass_type == 702:
                meta_scale = 500.0
            ratio = nn.functional.sigmoid(scale * meta_scale)

            self._ratio.requires_grad = False
            self._ratio.data = self._ratio.data.detach()
            self._ratio.data[idx] = ratio.detach()
            
            result = og_hidden_states * (1.0 - ratio) + hidden_states * ratio
                
            self._denominator.requires_grad = False
            self._denominator.data = self._denominator.data.detach()
            self._denominator.data[idx].fill_(1.0)

            return result.to(dtype=dtype)
        elif self.config.pass_type >= 800 and self.config.pass_type < 900:
            dtype = og_hidden_states.dtype
            temp_dtype = torch.float32
            factor = self.weight[idx].to(dtype=temp_dtype)
            og_hidden_states = og_hidden_states.to(dtype=temp_dtype)
            hidden_states = hidden_states.to(dtype=temp_dtype)

            self._ratio.requires_grad = False
            self._ratio.data = self._ratio.data.detach()

            if self.config.pass_type == 800:
                factor = factor.abs()
            elif self.config.pass_type == 801:
                factor = torch.relu(factor)

            alpha = factor / (factor + self.threshold)
            one_minus_alpha = 1.0 - alpha

            result = og_hidden_states * one_minus_alpha + hidden_states * alpha

            # these are just for diagnostics
            self._ratio.data[idx] = alpha.squeeze()
            self._denominator.requires_grad = False
            self._denominator.data = self._denominator.data.detach()
            self._denominator.data[idx] = (factor + self.threshold).squeeze()

            return result.to(dtype=dtype)
        else:
            return (
                og_hidden_states * self.threshold
                + hidden_states * self.weight[idx].abs()
            ) / (self.threshold + self.weight[idx].abs())


# Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM with LLAMA->GEMMA,Llama->Gemma,llama->gemma
class ThinkGemmaForCausalLM(GemmaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

        if config.pass_type >= 3:
            self.pass_scale_k = PassScale(config)
            self.pass_scale_v = PassScale(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # Ignore copy
    @add_start_docstrings_to_model_forward(GEMMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor,
        prompt_lengths: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        enforce_bidir: Optional[bool] = False,
        call_depth: Optional[int] = 0,
        peft_model=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GemmaForCausalLM

        >>> model = GemmaForCausalLM.from_pretrained("google/gemma-7b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""

        if self.config.pass_type > 0 and call_depth == 0 and cache_position is None:
            return _pass_fn(
                peft_model,
                input_ids=input_ids,
                prompt_lengths=prompt_lengths,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                enforce_bidir=enforce_bidir,
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            enforce_bidir=enforce_bidir,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens - self.model.think_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = (
                    past_key_values[0][0].shape[2] - self.model.think_tokens
                )
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if self.generation_config.cache_implementation == "static":
            # generation with static cache
            cache_position = kwargs.get("cache_position", None)
            if cache_position is None:
                past_length = 0
            else:
                past_length = cache_position[-1] + 1
            input_ids = input_ids[:, past_length:]
            position_ids = position_ids[:, past_length:]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = torch.arange(
            past_length,
            past_length + position_ids.shape[-1],
            device=position_ids.device,
        )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids.contiguous(),
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "prompt_lengths": kwargs.get("prompt_lengths"),
                "peft_model": (
                    kwargs.get("peft_model") if "peft_model" in kwargs else None
                ),
            }
        )
        # print(model_inputs["position_ids"], model_inputs["cache_position"])
        model_inputs["position_ids"] = None
        if model_inputs["cache_position"][0] == 0:
            model_inputs["cache_position"] = None
        else:
            model_inputs["cache_position"][0] = (
                model_inputs["cache_position"][0] + self.model.think_tokens
            )
        # model_inputs["use_cache"] = False
        # model_inputs["cache_position"] = None
        # model_inputs["past_key_values"] = None
        # model_inputs["attention_mask"] = None
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
