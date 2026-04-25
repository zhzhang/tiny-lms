"""Comprehensive tests for my_sdpa against torch.nn.functional.scaled_dot_product_attention."""

import math

import pytest
import torch
import torch.nn.functional as F

from sdpa_test import my_sdpa

SEED = 42


def _ref(q, k, v, **kwargs):
    """Reference: PyTorch's built-in SDPA (no dropout for determinism)."""
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


def _assert_close(actual, expected, atol=1e-5, rtol=1e-5):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Basic shapes
# ---------------------------------------------------------------------------


class TestBasicShapes:
    """Q, K, V with matching dims, no optional args."""

    def test_4d(self):
        """Standard (batch, heads, seq, embed)."""
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        _assert_close(my_sdpa(q, k, v), _ref(q, k, v))

    def test_3d(self):
        """(heads, seq, embed) — no batch dim."""
        q = torch.randn(4, 8, 16)
        k = torch.randn(4, 8, 16)
        v = torch.randn(4, 8, 16)
        _assert_close(my_sdpa(q, k, v), _ref(q, k, v))

    def test_2d(self):
        """(seq, embed) — bare minimum."""
        q = torch.randn(8, 16)
        k = torch.randn(8, 16)
        v = torch.randn(8, 16)
        _assert_close(my_sdpa(q, k, v), _ref(q, k, v))

    def test_5d(self):
        """Extra batch dimensions: (B1, B2, H, L, E)."""
        q = torch.randn(2, 3, 4, 8, 16)
        k = torch.randn(2, 3, 4, 8, 16)
        v = torch.randn(2, 3, 4, 8, 16)
        _assert_close(my_sdpa(q, k, v), _ref(q, k, v))

    def test_single_element(self):
        """Degenerate case: L=S=1, E=1."""
        q = torch.randn(1, 1, 1, 1)
        k = torch.randn(1, 1, 1, 1)
        v = torch.randn(1, 1, 1, 1)
        _assert_close(my_sdpa(q, k, v), _ref(q, k, v))


# ---------------------------------------------------------------------------
# Asymmetric shapes
# ---------------------------------------------------------------------------


class TestAsymmetricShapes:
    """Q and K can differ in seq length; V can differ in embed dim."""

    def test_different_seq_lengths(self):
        """L=5 queries attending over S=10 keys/values."""
        q = torch.randn(2, 4, 5, 16)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        out = my_sdpa(q, k, v)
        _assert_close(out, _ref(q, k, v))
        assert out.shape == (2, 4, 5, 16)

    def test_different_value_embed_dim(self):
        """Ev != E — value vectors may have a different width."""
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 32)
        out = my_sdpa(q, k, v)
        _assert_close(out, _ref(q, k, v))
        assert out.shape == (2, 4, 8, 32)

    def test_different_seq_and_value_dim(self):
        """L != S *and* Ev != E simultaneously."""
        q = torch.randn(2, 4, 5, 16)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 32)
        out = my_sdpa(q, k, v)
        _assert_close(out, _ref(q, k, v))
        assert out.shape == (2, 4, 5, 32)

    def test_long_query_short_kv(self):
        """L > S."""
        q = torch.randn(2, 4, 20, 16)
        k = torch.randn(2, 4, 5, 16)
        v = torch.randn(2, 4, 5, 16)
        out = my_sdpa(q, k, v)
        _assert_close(out, _ref(q, k, v))
        assert out.shape == (2, 4, 20, 16)


# ---------------------------------------------------------------------------
# Attention masks
# ---------------------------------------------------------------------------


class TestAttnMask:
    """Boolean and float attention masks in various shapes."""

    def test_bool_mask_full_true(self):
        """All-True mask — same as no mask."""
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        mask = torch.ones(8, 8, dtype=torch.bool)
        _assert_close(
            my_sdpa(q, k, v, attn_mask=mask),
            _ref(q, k, v, attn_mask=mask),
        )

    def test_bool_mask_lower_triangular(self):
        """Causal-style bool mask."""
        q = torch.randn(2, 4, 6, 16)
        k = torch.randn(2, 4, 6, 16)
        v = torch.randn(2, 4, 6, 16)
        mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
        _assert_close(
            my_sdpa(q, k, v, attn_mask=mask),
            _ref(q, k, v, attn_mask=mask),
        )

    def test_bool_mask_block_diagonal(self):
        """Arbitrary pattern: block diagonal."""
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[:2, :2] = True
        mask[2:, 2:] = True
        _assert_close(
            my_sdpa(q, k, v, attn_mask=mask),
            _ref(q, k, v, attn_mask=mask),
        )

    def test_float_mask_zeros(self):
        """All-zero float mask — same as no mask."""
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        mask = torch.zeros(8, 8)
        _assert_close(
            my_sdpa(q, k, v, attn_mask=mask),
            _ref(q, k, v, attn_mask=mask),
        )

    def test_float_mask_neginf(self):
        """Float mask with -inf to suppress certain positions."""
        q = torch.randn(2, 4, 6, 16)
        k = torch.randn(2, 4, 6, 16)
        v = torch.randn(2, 4, 6, 16)
        mask = torch.zeros(6, 6)
        mask[0, 3:] = float("-inf")
        mask[1, 4:] = float("-inf")
        _assert_close(
            my_sdpa(q, k, v, attn_mask=mask),
            _ref(q, k, v, attn_mask=mask),
        )

    def test_float_mask_additive_bias(self):
        """Float mask with arbitrary additive values (not just 0/-inf)."""
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        mask = torch.randn(4, 4)
        _assert_close(
            my_sdpa(q, k, v, attn_mask=mask),
            _ref(q, k, v, attn_mask=mask),
        )

    def test_mask_broadcastable_4d(self):
        """Mask shape (1, 1, L, S) broadcast over batch and heads."""
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        _assert_close(
            my_sdpa(q, k, v, attn_mask=mask),
            _ref(q, k, v, attn_mask=mask),
        )

    def test_mask_per_head(self):
        """Different mask per head: (1, H, L, S)."""
        q = torch.randn(2, 4, 6, 16)
        k = torch.randn(2, 4, 6, 16)
        v = torch.randn(2, 4, 6, 16)
        mask = torch.randn(1, 4, 6, 6)
        _assert_close(
            my_sdpa(q, k, v, attn_mask=mask),
            _ref(q, k, v, attn_mask=mask),
        )

    def test_mask_per_batch(self):
        """Different mask per batch element: (B, 1, L, S)."""
        q = torch.randn(3, 4, 6, 16)
        k = torch.randn(3, 4, 6, 16)
        v = torch.randn(3, 4, 6, 16)
        mask = torch.randn(3, 1, 6, 6)
        _assert_close(
            my_sdpa(q, k, v, attn_mask=mask),
            _ref(q, k, v, attn_mask=mask),
        )

    def test_mask_asymmetric_seq(self):
        """Mask when L != S."""
        q = torch.randn(2, 4, 5, 16)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        mask = torch.ones(5, 10, dtype=torch.bool)
        mask[:, 8:] = False
        _assert_close(
            my_sdpa(q, k, v, attn_mask=mask),
            _ref(q, k, v, attn_mask=mask),
        )


# ---------------------------------------------------------------------------
# Causal masking
# ---------------------------------------------------------------------------


class TestCausal:
    """is_causal=True generates a lower-triangular mask."""

    def test_causal_square(self):
        """L == S — standard causal."""
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        _assert_close(
            my_sdpa(q, k, v, is_causal=True),
            _ref(q, k, v, is_causal=True),
        )

    def test_causal_nonsquare(self):
        """L < S — upper-left causal bias."""
        q = torch.randn(2, 4, 5, 16)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        _assert_close(
            my_sdpa(q, k, v, is_causal=True),
            _ref(q, k, v, is_causal=True),
        )

    def test_causal_seq_len_1(self):
        """Single query token — causal mask is trivially all-attend."""
        q = torch.randn(2, 4, 1, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        _assert_close(
            my_sdpa(q, k, v, is_causal=True),
            _ref(q, k, v, is_causal=True),
        )

    def test_causal_matches_manual_tril_mask(self):
        """Causal output should equal manually applying a tril bool mask."""
        q = torch.randn(2, 4, 6, 16)
        k = torch.randn(2, 4, 6, 16)
        v = torch.randn(2, 4, 6, 16)
        causal_out = my_sdpa(q, k, v, is_causal=True)
        tril_mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
        mask_out = _ref(q, k, v, attn_mask=tril_mask)
        _assert_close(causal_out, mask_out)

    def test_causal_3d(self):
        """Causal with 3D input."""
        q = torch.randn(4, 8, 16)
        k = torch.randn(4, 8, 16)
        v = torch.randn(4, 8, 16)
        _assert_close(
            my_sdpa(q, k, v, is_causal=True),
            _ref(q, k, v, is_causal=True),
        )


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


class TestScale:
    """Custom scale factor (keyword-only arg)."""

    def test_custom_scale(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        _assert_close(
            my_sdpa(q, k, v, scale=0.5),
            _ref(q, k, v, scale=0.5),
        )

    def test_scale_one(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        _assert_close(
            my_sdpa(q, k, v, scale=1.0),
            _ref(q, k, v, scale=1.0),
        )

    def test_scale_very_small(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        _assert_close(
            my_sdpa(q, k, v, scale=1e-4),
            _ref(q, k, v, scale=1e-4),
        )

    def test_default_scale_equals_inv_sqrt_E(self):
        """Default scale should be 1/sqrt(E)."""
        q = torch.randn(1, 1, 4, 64)
        k = torch.randn(1, 1, 4, 64)
        v = torch.randn(1, 1, 4, 64)
        _assert_close(
            my_sdpa(q, k, v),
            my_sdpa(q, k, v, scale=1.0 / math.sqrt(64)),
        )

    def test_scale_with_causal(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        _assert_close(
            my_sdpa(q, k, v, is_causal=True, scale=0.1),
            _ref(q, k, v, is_causal=True, scale=0.1),
        )

    def test_scale_with_mask(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        mask = torch.randn(8, 8)
        _assert_close(
            my_sdpa(q, k, v, attn_mask=mask, scale=2.0),
            _ref(q, k, v, attn_mask=mask, scale=2.0),
        )


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------


class TestGQA:
    """enable_gqa=True — query heads are a multiple of key/value heads."""

    def test_gqa_basic(self):
        """8 query heads, 4 KV heads."""
        q = torch.randn(2, 8, 10, 16)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        _assert_close(
            my_sdpa(q, k, v, enable_gqa=True),
            _ref(q, k, v, enable_gqa=True),
        )

    def test_gqa_single_kv_head(self):
        """Multi-query attention: 1 KV head shared by all query heads."""
        q = torch.randn(2, 8, 10, 16)
        k = torch.randn(2, 1, 10, 16)
        v = torch.randn(2, 1, 10, 16)
        _assert_close(
            my_sdpa(q, k, v, enable_gqa=True),
            _ref(q, k, v, enable_gqa=True),
        )

    def test_gqa_same_heads(self):
        """Hq == H — GQA degenerates to standard MHA."""
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        out_gqa = my_sdpa(q, k, v, enable_gqa=True)
        out_std = my_sdpa(q, k, v, enable_gqa=False)
        _assert_close(out_gqa, out_std)

    def test_gqa_with_causal(self):
        q = torch.randn(2, 8, 10, 16)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        _assert_close(
            my_sdpa(q, k, v, is_causal=True, enable_gqa=True),
            _ref(q, k, v, is_causal=True, enable_gqa=True),
        )

    def test_gqa_with_mask(self):
        q = torch.randn(2, 8, 10, 16)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        mask = torch.ones(10, 10, dtype=torch.bool).tril()
        _assert_close(
            my_sdpa(q, k, v, attn_mask=mask, enable_gqa=True),
            _ref(q, k, v, attn_mask=mask, enable_gqa=True),
        )

    def test_gqa_with_scale(self):
        q = torch.randn(2, 8, 10, 16)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        _assert_close(
            my_sdpa(q, k, v, scale=0.25, enable_gqa=True),
            _ref(q, k, v, scale=0.25, enable_gqa=True),
        )

    def test_gqa_different_kv_heads_for_k_and_v(self):
        """K and V may have different head counts (both divide Hq)."""
        q = torch.randn(2, 12, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 3, 8, 16)
        _assert_close(
            my_sdpa(q, k, v, enable_gqa=True),
            _ref(q, k, v, enable_gqa=True),
        )

    def test_gqa_llama3_style(self):
        """Llama-3-style: 32 query heads, 8 KV heads."""
        q = torch.randn(1, 32, 16, 64)
        k = torch.randn(1, 8, 16, 64)
        v = torch.randn(1, 8, 16, 64)
        _assert_close(
            my_sdpa(q, k, v, enable_gqa=True),
            _ref(q, k, v, enable_gqa=True),
        )


# ---------------------------------------------------------------------------
# Dropout (shape-only — values are non-deterministic)
# ---------------------------------------------------------------------------


class TestDropout:
    """Dropout can't be tested for exact values; verify shape & no crash."""

    def test_dropout_nonzero(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        out = my_sdpa(q, k, v, dropout_p=0.1)
        assert out.shape == (2, 4, 8, 16)

    def test_dropout_zero_matches_ref(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        _assert_close(
            my_sdpa(q, k, v, dropout_p=0.0),
            _ref(q, k, v, dropout_p=0.0),
        )

    def test_dropout_half(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        out = my_sdpa(q, k, v, dropout_p=0.5)
        assert out.shape == (2, 4, 8, 16)


# ---------------------------------------------------------------------------
# Combined kwargs
# ---------------------------------------------------------------------------


class TestCombinations:
    """Multiple optional args used together."""

    def test_causal_and_scale(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        _assert_close(
            my_sdpa(q, k, v, is_causal=True, scale=0.3),
            _ref(q, k, v, is_causal=True, scale=0.3),
        )

    def test_mask_and_scale(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        mask = torch.tril(torch.ones(8, 8, dtype=torch.bool))
        _assert_close(
            my_sdpa(q, k, v, attn_mask=mask, scale=0.2),
            _ref(q, k, v, attn_mask=mask, scale=0.2),
        )

    def test_gqa_causal_scale(self):
        q = torch.randn(2, 8, 10, 16)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        _assert_close(
            my_sdpa(q, k, v, is_causal=True, scale=0.1, enable_gqa=True),
            _ref(q, k, v, is_causal=True, scale=0.1, enable_gqa=True),
        )

    def test_asymmetric_with_mask_and_scale(self):
        q = torch.randn(2, 4, 5, 16)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 32)
        mask = torch.randn(5, 10)
        _assert_close(
            my_sdpa(q, k, v, attn_mask=mask, scale=0.5),
            _ref(q, k, v, attn_mask=mask, scale=0.5),
        )


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestOutputShape:
    """Verify output tensor shapes across configurations."""

    @pytest.mark.parametrize(
        "q_shape, k_shape, v_shape, expected",
        [
            ((2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 16)),
            ((2, 4, 5, 16), (2, 4, 10, 16), (2, 4, 10, 32), (2, 4, 5, 32)),
            ((4, 8, 16), (4, 8, 16), (4, 8, 16), (4, 8, 16)),
            ((8, 16), (8, 16), (8, 16), (8, 16)),
            ((8, 16), (12, 16), (12, 32), (8, 32)),
            ((2, 3, 4, 8, 16), (2, 3, 4, 8, 16), (2, 3, 4, 8, 16), (2, 3, 4, 8, 16)),
        ],
    )
    def test_shapes(self, q_shape, k_shape, v_shape, expected):
        q = torch.randn(*q_shape)
        k = torch.randn(*k_shape)
        v = torch.randn(*v_shape)
        assert my_sdpa(q, k, v).shape == expected


# ---------------------------------------------------------------------------
# Numerical properties
# ---------------------------------------------------------------------------


class TestNumericalProperties:
    """Sanity-check numerical behavior."""

    def test_attention_weights_sum_to_one(self):
        """Each query's attention distribution should sum to ~1 (softmax)."""
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 6, 8)
        v_identity = torch.zeros(1, 1, 6, 1)
        v_identity[..., :, 0] = 1.0
        out = my_sdpa(q, k, v_identity)
        torch.testing.assert_close(out.squeeze(), torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_identical_keys_uniform_attention(self):
        """If all keys are identical, attention should be uniform."""
        q = torch.randn(1, 1, 3, 8)
        k = torch.ones(1, 1, 4, 8)
        v = torch.arange(4).float().view(1, 1, 4, 1).expand(1, 1, 4, 8)
        out = my_sdpa(q, k, v)
        expected_val = v.mean(dim=-2, keepdim=True).expand_as(out)
        _assert_close(out, expected_val, atol=1e-4, rtol=1e-4)

    def test_causal_first_row_only_sees_first_key(self):
        """With causal masking, first query attends only to first key."""
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        out = my_sdpa(q, k, v, is_causal=True)
        _assert_close(out[0, 0, 0], v[0, 0, 0])


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    """Verify that invalid inputs raise errors (matching PyTorch behavior)."""

    def _assert_raises_runtime_not_notimpl(self, fn):
        """Ensure a RuntimeError is raised, but NOT NotImplementedError."""
        with pytest.raises(AssertionError) as exc_info:
            fn()
        assert not isinstance(exc_info.value, NotImplementedError), (
            "Got NotImplementedError — implement the function first"
        )

    def test_1d_inputs(self):
        """Inputs must be at least 2D."""
        q = torch.randn(16)
        k = torch.randn(16)
        v = torch.randn(16)
        self._assert_raises_runtime_not_notimpl(lambda: my_sdpa(q, k, v))

    def test_query_key_embed_dim_mismatch(self):
        """Last dim of Q and K must match."""
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 32)
        v = torch.randn(2, 4, 8, 32)
        self._assert_raises_runtime_not_notimpl(lambda: my_sdpa(q, k, v))

    def test_batch_dim_mismatch(self):
        """Non-broadcastable batch dimensions."""
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(3, 4, 8, 16)
        v = torch.randn(3, 4, 8, 16)
        self._assert_raises_runtime_not_notimpl(lambda: my_sdpa(q, k, v))

    def test_gqa_non_divisible_heads(self):
        """Hq must be divisible by H when enable_gqa=True."""
        q = torch.randn(2, 7, 8, 16)
        k = torch.randn(2, 3, 8, 16)
        v = torch.randn(2, 3, 8, 16)
        self._assert_raises_runtime_not_notimpl(
            lambda: my_sdpa(q, k, v, enable_gqa=True)
        )

    def test_gqa_disabled_head_mismatch(self):
        """Without GQA, head dims must match (not broadcastable)."""
        q = torch.randn(2, 8, 10, 16)
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        self._assert_raises_runtime_not_notimpl(
            lambda: my_sdpa(q, k, v, enable_gqa=False)
        )


# ---------------------------------------------------------------------------
# Dtype handling
# ---------------------------------------------------------------------------


class TestDtypes:
    """Verify behavior across float dtypes."""

    def test_float32(self):
        q = torch.randn(2, 4, 8, 16, dtype=torch.float32)
        k = torch.randn(2, 4, 8, 16, dtype=torch.float32)
        v = torch.randn(2, 4, 8, 16, dtype=torch.float32)
        _assert_close(my_sdpa(q, k, v), _ref(q, k, v))

    def test_float64(self):
        q = torch.randn(2, 4, 8, 16, dtype=torch.float64)
        k = torch.randn(2, 4, 8, 16, dtype=torch.float64)
        v = torch.randn(2, 4, 8, 16, dtype=torch.float64)
        _assert_close(my_sdpa(q, k, v), _ref(q, k, v))
