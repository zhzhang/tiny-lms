import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    PRINTED = False

    def __init__(self, d_model: int, n_heads: int, context_length: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        # Reason for the above assert is detailed in the forward pass.
        self.n_heads = n_heads
        self.d_model = d_model
        # TODO: see what happens if we don't project in multi head attention.

        # Because in self attention Q, K, V are all just equal to the input x, we can project them all at once.
        self.input_projection = nn.Linear(d_model, 3 * d_model)

        self.output_projection = nn.Linear(d_model, d_model)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length)).view(
                1, 1, context_length, context_length
            ),
        )
        self.cache = None


    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        # X shape is (batch_size, sequence_length, d_model)
        # (batch_size, sequence_length, 3 * d_model)
        x = self.input_projection(x)
        # Each of q, k, v is (batch_size, sequence_length, n_dim)
        q, k, v = x.split(self.d_model, dim=2)
        if self.cache:
            prev_k, prev_v = self.cache
            k = torch.concat([prev_k, k], dim=1)
            v = torch.concat([prev_v, v], dim=1)
        batch_size, sequence_length, d_model = k.size()
        query_length = 1 if self.cache else sequence_length
        self.cache = (k, v)
        # Allocate slices of q, k, v for each head.
        # A linear map from d_input -> d_head is the same as a linear map from
        # d_input -> d_input because that linear map is just a stack of matrices
        # of size d_input x (d_input / n_heads) so long as there is divisibility.
        # Each of q, k, v is now (batch_size, sequence_length, n_heads, d_head)
        # where d_head = d_model / n_heads
        q = q.view(batch_size, query_length, self.n_heads, d_model // self.n_heads)
        k = k.view(batch_size, sequence_length, self.n_heads, d_model // self.n_heads)
        v = v.view(batch_size, sequence_length, self.n_heads, d_model // self.n_heads)
        # Swap the sequence length and head dimensions, as we are trying to end up with a sequence_length x sequence_length matrix.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # (batch_size, n_heads, sequence_length, sequence_length)
        attn = q @ k.transpose(-2, -1)
        # Scale the attention scores by 1 / sqrt(d_model)
        # TODO: what happens if we put the scale after the mask?
        attn = attn / math.sqrt(d_model // self.n_heads)
        if query_length > 1:
            attn = attn.masked_fill(
                self.mask[:, :, :sequence_length, :sequence_length] == 0, float("-inf")
            )
        # TODO: see if this softmax is in the right dimension.
        attn = attn.softmax(dim=-1)
        # (batch_size, n_heads, sequence_length, d_head)
        attn = attn @ v
        # Swap back the sequence length and head dimensions.
        output = (
            attn.transpose(1, 2).contiguous().view(batch_size, query_length, d_model)
        )
        output = self.output_projection(output)
        return output


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


class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.input = nn.Linear(d_model, 4 * d_model)
        self.gelu = NewGELU()
        self.output = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.gelu(x)
        x = self.output(x)
        return x


class Block(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, context_length: int
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.attention = AttentionHead(d_model, n_heads, context_length)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: be careful, the residual connection links the input pre layer norm, not post layer norm.
        x = self.attention(self.layer_norm_1(x)) + x
        x = self.feed_forward(self.layer_norm_2(x)) + x
        return x


class GPT(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, context_length: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.context_length = context_length

        self.token_embedding = nn.Embedding(50257, self.d_model)
        self.position_embedding = nn.Embedding(self.context_length, self.d_model)
        self.blocks = nn.ModuleList(
            [
                Block(self.d_model, self.n_heads, self.context_length)
                for _ in range(self.n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

        # Ties the token embedding and the output projection.
        # Output of the last transformer layer looks at the entire vocabulary and
        # picks via dot product which token is most likely to come next.
        self.lm_head = nn.Linear(self.d_model, 50257, bias=False)
        self.token_embedding.weight = self.lm_head.weight

        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)
        self.prefilled = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = (
                0.02
                if not hasattr(module, "LLMC_RESIDUAL_SCALE_FLAG")
                else 0.02 / math.sqrt(2 * self.config.n_layer)
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
        if self.prefilled:
            x = x[:, -1:]

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

        self.prefilled = True

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

    @classmethod
    def from_pretrained(cls, _sdf=None):
        """Loads pretrained GPT-2 model weights from huggingface"""
        from transformers import GPT2LMHeadModel

        print("mine: loading weights from pretrained gpt2 base")

        model = GPT(n_layers=12, n_heads=12, d_model=768, context_length=1024)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attention.mask")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        pretrained_model = GPT2LMHeadModel.from_pretrained("gpt2")
        pretrained_sd = pretrained_model.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        pretrained_sd_keys = pretrained_sd.keys()
        # ignore mask buffers
        pretrained_sd_keys = [
            k for k in pretrained_sd_keys if not k.endswith(".attn.masked_bias")
        ]
        pretrained_sd_keys = [
            k for k in pretrained_sd_keys if not k.endswith(".attn.bias")
        ]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        parameter_name_mapping = {}
        for k in sd_keys:
            target_key = None
            first, rest = k.split(".", 1)
            if first == "token_embedding":
                target_key = "transformer.wte.weight"
            elif first == "position_embedding":
                target_key = "transformer.wpe.weight"
            elif first == "blocks":
                idx, rest = rest.split(".", 1)
                if rest == "layer_norm_1.weight":
                    target_key = f"transformer.h.{idx}.ln_1.weight"
                elif rest == "layer_norm_1.bias":
                    target_key = f"transformer.h.{idx}.ln_1.bias"
                elif rest == "layer_norm_2.weight":
                    target_key = f"transformer.h.{idx}.ln_2.weight"
                elif rest == "layer_norm_2.bias":
                    target_key = f"transformer.h.{idx}.ln_2.bias"
                elif rest == "attention.input_projection.weight":
                    target_key = f"transformer.h.{idx}.attn.c_attn.weight"
                elif rest == "attention.input_projection.bias":
                    target_key = f"transformer.h.{idx}.attn.c_attn.bias"
                elif rest == "attention.output_projection.weight":
                    target_key = f"transformer.h.{idx}.attn.c_proj.weight"
                elif rest == "attention.output_projection.bias":
                    target_key = f"transformer.h.{idx}.attn.c_proj.bias"
                elif rest == "feed_forward.input.weight":
                    target_key = f"transformer.h.{idx}.mlp.c_fc.weight"
                elif rest == "feed_forward.input.bias":
                    target_key = f"transformer.h.{idx}.mlp.c_fc.bias"
                elif rest == "feed_forward.output.weight":
                    target_key = f"transformer.h.{idx}.mlp.c_proj.weight"
                elif rest == "feed_forward.output.bias":
                    target_key = f"transformer.h.{idx}.mlp.c_proj.bias"
            elif first == "layer_norm":
                if rest == "weight":
                    target_key = "transformer.ln_f.weight"
                elif rest == "bias":
                    target_key = "transformer.ln_f.bias"
            elif first == "lm_head":
                target_key = "lm_head.weight"
            if target_key is not None:
                parameter_name_mapping[k] = target_key
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(pretrained_sd_keys) == len(sd_keys) and len(
            pretrained_sd_keys
        ) == len(parameter_name_mapping), (
            f"mismatched keys: {len(pretrained_sd_keys)} != {len(sd_keys)}"
        )
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        for k in sd_keys:
            target_key = parameter_name_mapping[k]
            if any(target_key.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert pretrained_sd[target_key].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(pretrained_sd[target_key].t())
            else:
                # vanilla copy over the other parameters
                assert pretrained_sd[target_key].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(pretrained_sd[target_key])

        return model
