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
    assert len(query.shape) > 1 and len(key.shape) > 1 and len(value.shape) > 1
    *q_shapes, q_seq, q_dim = query.shape
    *k_shapes, k_seq, k_dim = key.shape
    assert q_dim == k_dim
    *v_shapes, v_seq, v_dim = value.shape

    no_extra_dims = len(q_shapes) == 0 and len(k_shapes) == 0 and len(v_shapes) == 0
    q_head = None if no_extra_dims else q_shapes[-1]
    k_head = None if no_extra_dims else k_shapes[-1]
    v_head = None if no_extra_dims else v_shapes[-1]

    assert no_extra_dims or (q_shapes[0] == v_shapes[0] and q_shapes[0] == k_shapes[0])

    assert k_seq == v_seq
    output_shape = (*q_shapes, q_seq, v_dim)

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

    if enable_gqa:
        assert q_head >= k_head and q_head % k_head == 0
        n_q_per_k = q_head // k_head
        query = query.reshape(*q_shapes[:-1], k_head, n_q_per_k, q_seq, q_dim)
        key = key.unsqueeze(-3)
    else:
        assert q_head == k_head and q_head == v_head

    attn = torch.einsum("...le,...se->...ls", query, key)
    attn *= scale if scale is not None else 1 / math.sqrt(q_dim)
    attn = attn + attn_bias
    attn = attn.softmax(-1)
    if enable_gqa:
        q_head = q_shapes[-1]
        v_head = v_shapes[-1]
        assert q_head >= v_head and q_head % v_head == 0
        n_q_per_v = q_head // v_head
        attn = attn.reshape(*q_shapes[:-1], v_head, n_q_per_v, q_seq, k_seq)
        value = value.unsqueeze(-3)
    out = torch.einsum("...qv,...ve->...qe", attn, value).reshape(output_shape)
    return out
