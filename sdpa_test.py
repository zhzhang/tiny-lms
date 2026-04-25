import math
from typing import Optional

import torch


def my_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Args:
        query:      (N, ..., Hq, L, E)
        key:        (N, ..., H,  S, E)
        value:      (N, ..., H,  S, Ev)
        attn_mask:  Broadcastable to (N, ..., L, S). Bool (True = attend) or
                    float (added to attention scores).
        dropout_p:  Dropout probability applied to attention weights.
        is_causal:  Apply lower-triangular causal mask. Must not be combined
                    with attn_mask.
        scale:      Scaling factor; defaults to 1/sqrt(E).  Keyword-only.
        enable_gqa: Grouped-query attention. Hq must be divisible by H.

    Returns:
        (N, ..., Hq, L, Ev)
    """
    *q_shapes, q_seq, q_dim = query.shape
    *k_shapes, k_seq, k_dim = key.shape
    n_dims = len(q_shapes)
    # if enable_gqa:
    #     assert q_head >= k_head and q_head % k_head == 0
    # else:
    #     assert q_head == k_head

    # Two no mask cases.
    if is_causal:
        assert attn_mask is None
        attn_mask = torch.ones(q_seq, k_seq, dtype=torch.bool).tril()
    if attn_mask is None:
        attn_mask = torch.ones(q_seq, k_seq, dtype=torch.bool)

    if attn_mask.dtype == torch.bool:
        attn_bias = torch.zeros(attn_mask.shape)
        attn_bias.masked_fill_(attn_mask.logical_not(), -float("inf"))
    else:
        attn_bias = attn_mask

    print(attn_mask)
    print(attn_bias)
    attn = torch.einsum("...le,...se->...ls", query, key)
    attn *= scale if scale is not None else 1 / math.sqrt(q_dim)
    attn = attn + attn_bias
    attn = attn.softmax(-1)
    out = torch.einsum("...qv,...ve->...qe", attn, value)
    return out
