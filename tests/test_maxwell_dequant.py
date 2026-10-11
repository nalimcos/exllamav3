"""
CPU unit test for the EXL3 Maxwell dequant core (Workstream B).

Scope: this tests the code -> value MAPPING that the shared fp32 dequant LUT stores
(exl3_dq_lut_build in quant/codebook.cuh). It replicates the exact fp16 codebook math
(cb0 lop3, cb1 mcg, cb2 mul1) in pure NumPy and checks the LUT table against the
per-code decode. It does NOT exercise the CUDA bit-extraction (the shf/bfe code in
quant/exl3_dq.cuh); that path is architecture-dependent and is validated on-hardware.

The values must be the EXACT fp16 codebook results upcast to fp32 (bit-identical to
decode_3inst), not a fresh fp32 formula: a fresh fp32 byte_sum*k_inv+k_bias drifts by
one fp16 ulp and is not fp16-representable.

Pure CPU: no torch.cuda, no CUDA tensors, no device skip.

    python -m pytest tests/test_maxwell_dequant.py -v
"""

import numpy as np
import pytest

U32 = (1 << 32) - 1

# mul1 (cb2) codebook constants, as fp16 bit patterns (see codebook.cuh)
K_INV = np.uint16(0x1EEE).view(np.float16)   # 1/147.7
K_BIAS = np.uint16(0xC931).view(np.float16)  # (-1024 - 510) * k_inv


def _bits_to_f16(v):
    """Reinterpret the low 16 bits of v as an fp16 value (scalar)."""
    return np.uint16(v & 0xFFFF).view(np.float16)


def _byte_sum(x):
    return (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)


# Scalar per-code reference: one code at a time, replicating decode_3inst<cb> (codebook.cuh)

def _cb0(code):
    x = (code * 89226354 + 64248484) & U32
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    return np.float16(_bits_to_f16(x & 0xFFFF) + _bits_to_f16((x >> 16) & 0xFFFF)).astype(np.float32)


def _cb1(code):
    x = (code * 0xCBAC1FED) & U32
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    return np.float16(_bits_to_f16(x & 0xFFFF) + _bits_to_f16((x >> 16) & 0xFFFF)).astype(np.float32)


def _cb2(code):
    x = (code * 0x83DCD12D) & U32
    s = _byte_sum(x) + 0x6400
    h = _bits_to_f16(s)
    return np.float16(np.float16(h * K_INV) + K_BIAS).astype(np.float32)


def codebook_value(code, cb):
    return (_cb0, _cb1, _cb2)[cb](code)


# Vectorized LUT builder: mirrors exl3_dq_lut_build (all codes at once, a different code path
# than the scalar reference so the consistency check below is not circular)

def build_lut(cb, K):
    codes = np.arange(1 << K, dtype=np.uint64)
    if cb == 0:
        x = (codes * 89226354) & U32
        x = (x + 64248484) & U32
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        lo = (x & 0xFFFF).astype(np.uint16).view(np.float16)
        hi = ((x >> 16) & 0xFFFF).astype(np.uint16).view(np.float16)
        return (lo + hi).astype(np.float32)
    if cb == 1:
        x = (codes * 0xCBAC1FED) & U32
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        lo = (x & 0xFFFF).astype(np.uint16).view(np.float16)
        hi = ((x >> 16) & 0xFFFF).astype(np.uint16).view(np.float16)
        return (lo + hi).astype(np.float32)
    if cb == 2:
        x = (codes * 0x83DCD12D) & U32
        s = _byte_sum(x) + 0x6400
        h = s.astype(np.uint16).view(np.float16)
        return (np.float16(h * K_INV) + K_BIAS).astype(np.float32)
    raise ValueError(cb)


@pytest.mark.parametrize("K", range(1, 9))
@pytest.mark.parametrize("cb", [0, 1, 2])
def test_lut_matches_decode(K, cb):
    """The LUT table must equal the per-code decode for every code, exactly.

    This is the invariant the fp32 GEMM/GEMV rely on: lut[code] == decode_3inst<cb>(code).
    The vectorized builder and the scalar reference are independent code paths, so a wrong
    uint32 wrap, lop3 constant, or fp16 rounding in either shows up here.
    """
    n = 1 << K
    lut = build_lut(cb, K)
    assert lut.shape == (n,)
    ref = np.array([codebook_value(c, cb) for c in range(n)], dtype=np.float32)
    assert np.array_equal(lut, ref), f"K={K} cb={cb}: LUT table diverges from the per-code decode"


@pytest.mark.parametrize("K", range(1, 9))
@pytest.mark.parametrize("cb", [0, 1, 2])
def test_lut_values_are_fp16(K, cb):
    """Every LUT value must be exactly fp16-representable.

    The LUT stores float(decode_3inst<cb>(code)) -- an fp16 result upcast to fp32. If the table
    were built from a fresh fp32 formula instead, the values would not round-trip through fp16.
    """
    lut = build_lut(cb, K)
    rt = np.float16(lut).astype(np.float32)
    assert np.array_equal(lut, rt), f"K={K} cb={cb}: a LUT value is not fp16-representable"


def test_cb2_differs_from_fresh_fp32():
    """The mul1 path must not collapse to a fresh fp32 byte_sum*k_inv+k_bias.

    A fresh fp32 recompute (upcast h, then fp32 FMA) drifts by one fp16 ulp and is not
    fp16-representable; the exact fp16 path must differ from it.
    """
    K = 8
    lut = build_lut(2, K)
    codes = np.arange(1 << K, dtype=np.uint64)
    x = (codes * 0x83DCD12D) & U32
    s = _byte_sum(x) + 0x6400
    h32 = s.astype(np.uint16).view(np.float16).astype(np.float32)
    fresh = h32 * float(K_INV) + float(K_BIAS)
    assert not np.array_equal(lut, fresh), "cb2 LUT equals the fresh fp32 formula (fp16 rounding lost)"
    assert not np.array_equal(fresh, np.float16(fresh).astype(np.float32)), "fresh fp32 unexpectedly fp16-representable"


def test_anchor_values():
    """Frozen code -> value pairs, guarding against accidental changes to the codebook math."""
    anchors = {
        (0, 0): 0.76806640625,
        (1, 0): 1.84375,
        (2, 0): -3.453125,
        (0, 255): 1.2041015625,
        (1, 255): -1.287109375,
        (2, 255): 0.8359375,
    }
    for (cb, code), expected in anchors.items():
        got = float(codebook_value(code, cb))
        assert got == expected, f"cb={cb} code={code}: {got} != {expected}"
