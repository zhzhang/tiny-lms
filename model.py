import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int
    n_kv_heads: int
    n_q_heads: int
    n_layers: int
    context_length: int
    vocab_size: int


class AttentionHead(nn.Module):
    PRINTED = False

    def __init__(self, config: ModelConfig):
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
            config.d_model, config.d_model + 2 * self.kv_dim
        )

        self.output_projection = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        # (batch_size, n_heads, sequence_length, sequence_length)
        output = (
            y.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, config.d_model)
        )
        output = self.output_projection(output)
        return output


class SwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.input = nn.Linear(d_model, 2 * d_model)
        self.output = nn.Linear(d_model, d_model)

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
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model)
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
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.d_model)
        self.attention = AttentionHead(config)
        self.layer_norm_2 = nn.LayerNorm(config.d_model)
        # self.feed_forward = SwiGLU(config.d_model)
        self.feed_forward = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: be careful, the residual connection links the input pre layer norm, not post layer norm.
        x = self.attention(self.layer_norm_1(x)) + x
        x = self.feed_forward(self.layer_norm_2(x)) + x
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Ties the token embedding and the output projection.
        # Output of the last transformer layer looks at the entire vocabulary and
        # picks via dot product which token is most likely to come next.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

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
        return_logits: bool = True,
    ) -> torch.Tensor:
        token_embeds = self.token_embedding(idx)
        position_embeds = self.position_embedding(
            torch.arange(idx.size(1), dtype=torch.long, device=idx.device)
        )
        x = token_embeds + position_embeds

        for i, block in enumerate(self.blocks):
            x = block(x)
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
        n_kv_heads=12,
        n_q_heads=12,
        n_layers=12,
        context_length=1024,
        vocab_size=50257,
    )
    model = Model(config)
    model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-4,
        betas=(0.9, 0.95),
        zero_stage=0,
    )
