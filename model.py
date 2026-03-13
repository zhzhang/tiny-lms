import math
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from enum import Enum


class PositionEmbeddingType(Enum):
    LEARNED = "learned"
    ROPE = "rope"
    NOPE = "nope"


@dataclass
class ModelConfig:
    d_model: int
    n_kv_heads: int
    n_q_heads: int
    n_layers: int
    max_sequence_length: int
    vocab_size: int
    position_embedding_type: PositionEmbeddingType
    rope_skip_freq: int = 2
    rope_theta: float = 10000.0
    no_bias: bool = False


class AttentionHead(nn.Module):
    PRINTED = False

    def __init__(self, config: ModelConfig, rotary_embedding: RotaryEmbedding = None):
        super().__init__()
        assert config.d_model % config.n_q_heads == 0, (
            "d_model must be divisible by n_q_heads"
        )
        assert config.d_model % config.n_kv_heads == 0, (
            "d_model must be divisible by n_kv_heads"
        )
        assert config.n_q_heads % config.n_kv_heads == 0, (
            "n_q_heads must be divisible by n_kv_heads"
        )
        self.config = config
        self.d_head = config.d_model // config.n_q_heads
        self.kv_dim = config.n_kv_heads * self.d_head
        self.input_projection = nn.Linear(
            config.d_model,
            config.d_model + 2 * self.kv_dim,
            bias=not config.no_bias,
        )

        self.output_projection = nn.Linear(
            config.d_model, config.d_model, bias=not config.no_bias
        )
        self.rotary_embedding = rotary_embedding

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        config = self.config
        # X shape is (batch_size, sequence_length, d_model)
        batch_size, sequence_length, _ = x.size()
        # (batch_size, sequence_length, 3 * d_model)
        x = self.input_projection(x)
        q, k, v = x.split([config.d_model, self.kv_dim, self.kv_dim], dim=2)
        # Allocate slices of q, k, v for each head.
        q = q.view(
            batch_size,
            sequence_length,
            config.n_q_heads,
            self.d_head,
        )
        k = k.view(batch_size, sequence_length, config.n_kv_heads, self.d_head)
        v = v.view(batch_size, sequence_length, config.n_kv_heads, self.d_head)
        # Swap the sequence length and head dimensions, as we are trying to end up with a sequence_length x sequence_length matrix.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if self.rotary_embedding is not None:
            q, k = self.rotary_embedding(q, k)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=attn_mask is None,
            enable_gqa=True,
        )
        # (batch_size, n_heads, sequence_length, sequence_length)
        output = (
            y.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, config.d_model)
        )
        output = self.output_projection(output)
        return output


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, no_bias: bool = False):
        super().__init__()
        self.input = nn.Linear(d_model, 2 * d_model, bias=not no_bias)
        self.output = nn.Linear(d_model, d_model, bias=not no_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        x = self.output(x)
        return x


class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""

    def forward(self, input):
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.d_model, 4 * config.d_model, bias=not config.no_bias
        )
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(
            4 * config.d_model, config.d_model, bias=not config.no_bias
        )
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        rotary_embedding: RotaryEmbedding = None,
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.d_model)
        self.attention = AttentionHead(config, rotary_embedding)
        self.layer_norm_2 = nn.LayerNorm(config.d_model)
        # self.feed_forward = SwiGLU(config.d_model)
        self.feed_forward = MLP(config)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # NOTE: be careful, the residual connection links the input pre layer norm, not post layer norm.
        x = self.attention(self.layer_norm_1(x), attn_mask) + x
        x = self.feed_forward(self.layer_norm_2(x)) + x
        return x


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Warm up cache.
        self.get_rotary_embedding(config.max_sequence_length, torch.device("cpu"))

    def get_rotary_embedding(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_q_heads
            inv_freq = 1.0 / (
                self.config.rope_theta
                ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = (
                positions.sin()[None, None, :, :],
                positions.cos()[None, None, :, :],
            )
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = (
                q_.shape[-2],
                k_.shape[-2],
            )  # could be different if layer_past not None
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        if config.position_embedding_type == PositionEmbeddingType.LEARNED:
            self.position_embedding = nn.Embedding(
                config.max_sequence_length, config.d_model
            )
        self.rotary_embedding = RotaryEmbedding(config)
        # Skip rotary embedding every `rope_skip_freq` layers.
        self.blocks = nn.ModuleList(
            [
                Block(
                    config,
                    self.rotary_embedding
                    if (
                        config.position_embedding_type == PositionEmbeddingType.ROPE
                        and i % config.rope_skip_freq == 0
                    )
                    else None,
                )
                for i in range(1, config.n_layers + 1)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Ties the token embedding and the output projection.
        # Output of the last transformer layer looks at the entire vocabulary and
        # picks via dot product which token is most likely to come next.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)
        self._validate_no_linear_bias()

    def _validate_no_linear_bias(self):
        if not self.config.no_bias:
            return
        for module_name, module in self.named_modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                raise ValueError(
                    f"`config.no_bias=True` but linear layer `{module_name}` has bias."
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = (
                0.02
                if not hasattr(module, "LLMC_RESIDUAL_SCALE_FLAG")
                else 0.02 / math.sqrt(2 * self.config.n_layers)
            )
            if not hasattr(module, "LLMC_SKIP_INIT"):
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=std, generator=self.init_rng
                )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02, generator=self.init_rng
            )

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> torch.Tensor:
        x = self.token_embedding(idx)
        if self.config.position_embedding_type == PositionEmbeddingType.LEARNED:
            position_embeds = self.position_embedding(
                torch.arange(idx.size(1), dtype=torch.long, device=idx.device)
            )
            x = x + position_embeds

        for i, block in enumerate(self.blocks):
            x = block(x, attn_mask)
        x = self.layer_norm(x)

        loss = None
        logits = self.lm_head(x)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    def configure_optimizers(
        self, weight_decay, learning_rate, betas, zero_stage, device_type=None
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        print("using regular AdamW")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


if __name__ == "__main__":
    config = ModelConfig(
        d_model=768,
        n_kv_heads=3,
        n_q_heads=12,
        n_layers=14,
        max_sequence_length=1024,
        vocab_size=50257,
        position_embedding_type=PositionEmbeddingType.ROPE,
        rope_skip_freq=2,
    )
    model = Model(config)
    model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-4,
        betas=(0.9, 0.95),
        zero_stage=0,
    )
    x = torch.randint(0, config.vocab_size, (1, 1024))
    model(x)
