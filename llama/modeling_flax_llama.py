# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" Flax LLaMA model."""
import functools
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax

import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask, dot_product_attention_weights
from flax.linen.linear import PrecisionLike
from flax.traverse_util import flatten_dict, unflatten_dict

from optax import (
    l2_loss,
    softmax_cross_entropy_with_integer_labels,
    sigmoid_binary_cross_entropy,
)

# from ...modeling_flax_outputs import (
#     FlaxBaseModelOutputWithPast,
#     FlaxCausalLMOutputWithCrossAttentions,
#     FlaxSequenceClassifierOutput
# )
# from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
# from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# from .configuration_llama import LlamaConfig

from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutputWithPast,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSequenceClassifierOutput,
)
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import LlamaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


class FlaxLlamaRMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6
    dtype: Optional[jnp.dtype] = None
    param_dtype: Optional[jnp.dtype] = None

    def setup(self) -> None:
        """
        FlaxLlamaRMSNorm is equivalent to T5LayerNorm
        """
        self.weight = self.param(
            "weight", nn.initializers.ones, (self.hidden_size,), self.param_dtype
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight


class FlaxLlamaRotaryEmbedding(nn.Module):
    dim: int
    max_position_embeddings: int = 2048
    base: int = 10000
    param_dtype: Optional[jnp.dtype] = None

    def setup(self):
        self.inv_freq = 1.0 / (
            self.base ** (jnp.arange(0, self.dim, 2, jnp.float32) / self.dim)
        )

        # Build here to make `jax.jit` work.
        self.max_seq_len_cached = self.max_position_embeddings
        t = jnp.arange(
            self.max_seq_len_cached,
            dtype=self.inv_freq.dtype,
        )
        freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = lax.concatenate([freqs, freqs], dimension=1)
        self.cos_cached = lax.cos(emb)[:, :].astype(self.param_dtype)
        self.sin_cached = lax.sin(emb)[:, :].astype(self.param_dtype)

    def __call__(self, dtype: jnp.dtype, seq_len: int):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = jnp.arange(
                self.max_seq_len_cached,
                dtype=self.inv_freq.dtype,
            )
            freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
            emb = lax.concatenate([freqs, freqs], dimension=1)
            self.cos_cached = lax.cos(emb)[:, :].astype(dtype)
            self.sin_cached = lax.sin(emb)[:, :].astype(dtype)
        return (
            self.cos_cached[:seq_len, ...].astype(dtype),
            self.sin_cached[:seq_len, ...].astype(dtype),
        )


@jax.jit
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return lax.concatenate((-x2, x1), dimension=3)


@jax.jit
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # Take [bsz, seq_len] from [seq_len, dim] in axis 0 (seq_len)
    cos = jnp.take(cos, position_ids, axis=0)[..., None, :]  # [bsz, seq_len, 1, dim]
    sin = jnp.take(sin, position_ids, axis=0)[..., None, :]  # [bsz, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FlaxLlamaMLP(nn.Module):
    config: LlamaConfig
    hidden_act: str
    dtype: Optional[jnp.dtype]
    param_dtype: Optional[jnp.dtype]
    precision: Optional[PrecisionLike]

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        self.resid_dropout = None
        if hasattr(self.config, "resid_pdrop"):
            self.resid_dropout = nn.Dropout(rate=self.config.resid_pdrop)
        dense_creator = functools.partial(
            nn.Dense,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            precision=self.precision,
        )
        if self.hidden_act == "silu":
            self.gate_up_proj = dense_creator(features=2 * self.intermediate_size)
            self.down_proj = dense_creator(features=self.hidden_size)
        else:
            # hidden_size -> intermediate_size
            self.gate_proj = dense_creator(features=self.intermediate_size)
            # intermediate_size -> hidden_size
            self.down_proj = dense_creator(features=self.hidden_size)
            # hidden_size -> intermediate_size
            self.up_proj = dense_creator(features=self.intermediate_size)
        self.act_fn = ACT2FN[self.hidden_act]

    def __call__(self, x: jnp.ndarray, deterministic: bool = False):
        if self.hidden_act == "silu":
            gate_up_states = self.gate_up_proj(x).reshape(x.shape[:2] + (-1, 2))
            gate_states, up_states = jnp.split(self.gate_up_proj(x), 2, axis=-1)
            gate_states = gate_states.reshape(x.shape[:2] + (-1,))
            up_states = up_states.reshape(x.shape[:2] + (-1,))
            hidden_states = self.down_proj(self.act_fn(gate_states) * up_states)
        else:
            hidden_states = self.down_proj(
                self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            )
        if self.resid_dropout:
            hidden_states = self.dropout(hidden_states, deterministic)
        return hidden_states


class FlaxLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper

    Attributes:
      config: LlamaConfig from HuggingFace.
      dtype: the dtype of the computation
        (default: infer from inputs and params)
      param_dtype: the dtype passed to parameter initializers (default: float32)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      attention_weights_fn: dot_product_attention_weights or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
    """

    config: LlamaConfig
    dtype: Optional[jnp.dtype]
    param_dtype: Optional[jnp.dtype]
    precision: Optional[PrecisionLike]
    attention_weights_fn: Callable[..., Any] = dot_product_attention_weights
    qkv_dot_general: Callable[..., Any] = lax.dot_general
    out_dot_general: Callable[..., Any] = lax.dot_general

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        assert self.hidden_size % self.num_heads == 0, (
            f"Memory dimension ({self.hidden_size}) must be divisible by number of"
            f" heads ({self.num_heads})."
        )
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.kernel_init = jax.nn.initializers.normal(
            stddev=self.config.initializer_range
        )

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Alpa currently does not support sharding for layers of nn.DenseGeneral with multiple outputs
        # self.qkv_proj = nn.DenseGeneral(
        #     features=(3 * self.num_heads, self.head_dim),
        #     axis=-1,
        #     dtype=self.dtype,
        #     param_dtype=self.param_dtype,
        #     kernel_init=self.kernel_init,
        #     use_bias=False,
        #     precision=self.precision,
        # )
        #
        # self.o_proj = nn.DenseGeneral(
        #     features=self.hidden_size,
        #     axis=(-2, -1),
        #     kernel_init=self.kernel_init,
        #     dtype=self.dtype,
        #     param_dtype=self.param_dtype,
        #     use_bias=False,
        #     precision=self.precision,
        # )

        self.qkv_proj = nn.Dense(
            features=(3 * self.hidden_size),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            use_bias=False,
            precision=self.precision,
        )

        self.o_proj = nn.Dense(
            features=self.hidden_size,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            precision=self.precision,
        )

        self.rotary_emb = FlaxLlamaRotaryEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings
        )
        self.resid_dropout = None
        if hasattr(self.config, "resid_pdrop"):
            self.resid_dropout = nn.Dropout(rate=self.config.resid_pdrop)

    def _shape(self, tensor: jnp.ndarray, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    @nn.compact
    def _concatenate_to_cache(self, query, key, value, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable(
            "cache", "cached_key", jnp.zeros, key.shape, key.dtype
        )
        cached_value = self.variable(
            "cache", "cached_value", jnp.zeros, value.shape, value.dtype
        )
        cache_index = self.variable(
            "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
        )

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        deterministic: bool = False,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.reshape(
            qkv_states.shape[:2] + (-1, 3)
        )  # (bs, seq_len, hs * 3) -> (bs, seq_len, hs, 3) for performance
        query_states, key_states, value_states = jnp.split(
            qkv_states, 3, axis=-1
        )  # (bs, seq_len, hs, 3) -> 3 * (bs, seq_len, hs, 1)
        query_states = self._split_heads(
            query_states
        )  # (bs, seq_len, hs, 1) -> (bs, seq_len, nh, hs)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        # Alpa currently does not support sharding for layers of nn.DenseGeneral with multiple outputs
        # qkv_states = self.qkv_proj(hidden_states)
        # query_states, key_states, value_states = jnp.split(qkv_states, 3, axis=-2)

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") and use_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                query_states, key_states, value_states, attention_mask
            )

        dropout_rng = None
        att_dropout_rate = 0.0
        if hasattr(self.config, "attn_pdrop"):
            att_dropout_rate = self.config.attn_pdrop
            if (
                not deterministic and att_dropout_rate > 0.0
            ):  # Require `deterministic` only if using dropout.
                dropout_rng = self.make_rng("dropout")

        # apply rotary position embedding
        kv_seq_len = key_states.shape[1]
        cos, sin = self.rotary_emb(value_states.dtype, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        # [bsz, t, nh, hd]

        # transform boolean mask into float mask
        if attention_mask is not None:
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                    self.dtype
                ),
            )
        else:
            attention_bias = None

        # usual dot product attention
        attn_weights = self.attention_weights_fn(
            query_states,
            key_states,
            bias=attention_bias,
            mask=attention_mask,
            dropout_rng=dropout_rng,
            dropout_rate=att_dropout_rate,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision,
        )  # pytype: disable=wrong-keyword-args
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        # back to the original inputs dimensions
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)
        if self.resid_dropout:
            attn_output = self.resid_dropout(
                self.resid_dropout, deterministic=deterministic
            )

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class FlaxLlamaDecoderLayer(nn.Module):
    config: LlamaConfig
    dtype: Optional[jnp.dtype]
    param_dtype: Optional[jnp.dtype]
    precision: Optional[PrecisionLike]

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.self_attn = FlaxLlamaAttention(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.mlp = FlaxLlamaMLP(
            config=self.config,
            hidden_act=self.config.hidden_act,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.input_layernorm = FlaxLlamaRMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.post_attention_layernorm = FlaxLlamaRMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        deterministic: bool = False,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """
        Args:
            hidden_states (`jnp.ndarray`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`jnp.ndarray`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            deterministic=deterministic,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class FlaxLlamaDecoderLayerCollection(nn.Module):
    config: LlamaConfig
    dtype: Optional[jnp.dtype]
    param_dtype: Optional[jnp.dtype]
    precision: Optional[PrecisionLike]
    gradient_checkpointing: bool = False

    def setup(self):
        self.layers = [
            FlaxLlamaDecoderLayer(
                self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(i),
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        deterministic: bool = False,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing:
                layer_outputs = jax.checkpoint(
                    layer.__call__,
                    static_argnums=(3, 4, 5),
                )(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    deterministic,
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    deterministic=deterministic,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # this contains possible `None` values - `FlaxLlamaModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxLlamaModule(nn.Module):
    config: LlamaConfig
    dtype: Optional[jnp.dtype]
    param_dtype: Optional[jnp.dtype]
    precision: Optional[PrecisionLike]

    # Copied from flax.linen.attention.make_causal_mask
    def _prepare_decoder_attention_mask(
        self,
        attention_mask: jnp.ndarray,
        mask_dtype: jnp.dtype,
        extra_batch_dims: int = 0,
    ):
        """create causal mask
        In case of 1d inputs (i.e., `[batch..., len]`, the self-attention weights
        will be `[batch..., heads, len, len]` and this function will produce a
        causal mask of shape `[batch..., 1, len, len]`.
        [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        Assert seq_len==tgt_seq_len==src_seq_len

        Args:
            x: input array of shape `[batch..., len]`
            extra_batch_dims: number of batch dims to add singleton axes for,
            none by default
            dtype: mask return dtype

        Returns:
            A `[batch..., 1, len, len]` shaped causal mask for 1d attention.
        """
        combined_attention_mask = make_causal_mask(
            x=attention_mask, extra_batch_dims=extra_batch_dims, dtype=mask_dtype
        )
        return combined_attention_mask

    def setup(self):
        self.embed_dim = self.config.hidden_size

        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.embd_dropout = None
        if hasattr(self.config, "embd_pdrop"):
            self.embd_dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.layers_collection = FlaxLlamaDecoderLayerCollection(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.norm = FlaxLlamaRMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        input_ids: Optional[jnp.ndarray],
        attention_mask: Optional[jnp.ndarray],
        position_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
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

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if self.has_variable("cache", "cached_key") is True:
            past_key_values_length = self.get_variable("cache", "cache_index").value
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = jnp.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=jnp.int32,
            )
            position_ids = jnp.expand_dims(position_ids, 0).reshape(-1, seq_length)
        else:
            position_ids = position_ids.reshape(-1, seq_length).astype(jnp.int32)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype(jnp.int32))

        # embed positions
        if attention_mask is None:
            attention_mask = jnp.ones(inputs_embeds.shape[:2], dtype=jnp.int32)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, inputs_embeds.dtype
        )

        hidden_states = inputs_embeds
        if self.embd_dropout:
            hidden_states = self.embd_dropout(
                hidden_states, deterministic=deterministic
            )

        outputs = self.layers_collection(
            hidden_states,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[2],
        )


class FlaxLlamaLMHeadModule(nn.Module):
    config: LlamaConfig
    dtype: Optional[jnp.dtype]
    param_dtype: Optional[jnp.dtype]
    precision: Optional[PrecisionLike]

    def setup(self):
        self.transformer = FlaxLlamaModule(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
        )

    def __call__(
        self,
        input_ids: Optional[jnp.ndarray],
        attention_mask: Optional[jnp.ndarray],
        position_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            inputs_embeds,
            deterministic=deterministic,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["embed_tokens"][
                "embedding"
            ].T
            lm_logits = self.lm_head.apply(
                {"params": {"kernel": shared_kernel}}, hidden_states
            )
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxLlamaScoreModule(nn.Module):
    config: LlamaConfig
    dtype: Optional[jnp.dtype]
    param_dtype: Optional[jnp.dtype]
    precision: Optional[PrecisionLike]

    def setup(self):
        self.transformer = FlaxLlamaModule(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.num_labels = self.config.num_labels
        self.score = nn.Dense(
            self.num_labels,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
        )

    def __call__(
        self,
        input_ids: Optional[jnp.ndarray],
        attention_mask: Optional[jnp.ndarray],
        position_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            inputs_embeds,
            deterministic=deterministic,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["embed_tokens"][
                "embedding"
            ].T
            lm_logits = self.lm_head.apply(
                {"params": {"kernel": shared_kernel}}, hidden_states
            )
        else:
            lm_logits = self.score(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


LLAMA_START_DOCSTRING = r"""    
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`LlamaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            self.config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `self.config.use_cache=True`):
            Tuple of `tuple(jnp.ndarray)` of length `self.config.n_layers`, 
            with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
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


class FlaxLlamaPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LlamaConfig
    base_model_prefix = "model"
    module_class: nn.Module = None
    supports_gradient_checkpointing = True
    _no_split_modules = ["FlaxLlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(
        self,
        config: LlamaConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float16,
        _do_init: bool = True,
        **kwargs,
    ):
        param_dtype = kwargs.get("param_dtype", dtype)
        precision = kwargs.get("precision", "fastest")
        module = self.module_class(
            config=config, dtype=dtype, param_dtype=param_dtype, precision=precision
        )
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None
    ) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype=jnp.int32)
        attention_mask = jnp.ones_like(input_ids)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(
            rngs, input_ids, attention_mask, return_dict=False
        )

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (FlaxLlamaModule)):
            module.layers_collection.gradient_checkpointing = value
        elif isinstance(module, (FlaxLlamaLMHeadModule)):
            module.transformer.layers_collection.gradient_checkpointing = value
        else:
            raise ValueError(
                "No such module %s could be set gradient checkpointing." % module
            )

    def use_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            input_ids,
            attention_mask,
            return_dict=False,
            use_cache=True,
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids: Optional[jnp.ndarray],
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        use_cache: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Any:
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
            return_dict if return_dict is not None else self.config.return_dict
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag use_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxLlamaAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            input_ids,
            attention_mask,
            position_ids,
            inputs_embeds,
            deterministic=not train,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        # outputs of module.apply return the outputs and a variable dict
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


@add_start_docstrings(
    "The bare LLaMA Model transformer outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class FlaxLlamaModel(FlaxLlamaPreTrainedModel):
    module_class = FlaxLlamaModule


@add_start_docstrings(
    "The bare LLaMA Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings.",
    LLAMA_START_DOCSTRING,
)
class FlaxLlamaForCausalLM(FlaxLlamaPreTrainedModel):
    module_class = FlaxLlamaLMHeadModule

    def get_input_embeddings(self):
        return self.module.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.module.transformer.embed_tokens = value

    def get_output_embeddings(self):
        return self.module.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.module.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.module = decoder

    def get_decoder(self):
        return self.module

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC
    )
    def __call__(
        self,
        input_ids: Optional[jnp.ndarray],
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        labels: Optional[jnp.ndarray] = None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FlaxCausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            labels (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                self.config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., self.config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = super(FlaxLlamaForCausalLM, self).__call__(
            input_ids,
            attention_mask,
            position_ids,
            inputs_embeds,
            params,
            past_key_values,
            dropout_rng,
            train,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = softmax_cross_entropy_with_integer_labels
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return FlaxCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: jnp.ndarray,
        past_key_values: Optional[dict],
        attention_mask: Optional[jnp.DeviceArray],
        inputs_embeds: Optional[jnp.DeviceArray],
        max_length: int,
        **kwargs,
    ):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        if past_key_values is None:
            past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since LLaMA uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = jnp.ones(
                (batch_size, max_length), dtype=jnp.int32
            )
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, attention_mask, (0, 0)
            )
        else:
            position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype=jnp.int32)[None, :],
                (batch_size, seq_length),
            )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class FlaxLlamaForSequenceClassification(FlaxLlamaPreTrainedModel):
    module_class = FlaxLlamaScoreModule

    def get_input_embeddings(self):
        return self.module.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.module.transformer.embed_tokens = value

    def get_output_embeddings(self):
        return self.module.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.module.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.module = decoder

    def get_decoder(self):
        return self.module

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC
    )
    def __call__(
        self,
        input_ids: Optional[jnp.ndarray],
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        labels: Optional[jnp.ndarray] = None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FlaxSequenceClassifierOutput]:
        r"""
        Args:
            labels (`jnp.ndarray` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                self.config.num_labels - 1]`. If `self.config.num_labels == 1` a regression loss is computed
                (Mean-Square loss), If `self.config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = super(FlaxLlamaForCausalLM, self).__call__(
            input_ids,
            attention_mask,
            position_ids,
            inputs_embeds,
            params,
            past_key_values,
            dropout_rng,
            train,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        logits = transformer_outputs.logits

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    jnp.not_equal(input_ids, self.config.pad_token_id).sum(-1) - 1
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[jnp.arange(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == jnp.int64 or labels.dtype == jnp.int32
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = l2_loss
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = softmax_cross_entropy_with_integer_labels
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = sigmoid_binary_cross_entropy()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return FlaxSequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
