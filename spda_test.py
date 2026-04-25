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
    attn = torch.einsum("...qle,...qle", query, key)
    scale = 
