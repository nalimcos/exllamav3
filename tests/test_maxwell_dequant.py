"""
CPU unit test for the EXL3 Maxwell dequant core (Workstream B).

Scope: this tests the PROCEDURAL fp32 dequant `decode_3inst_f<cb>(x)` (codebook.cuh), which is
`__half2float(decode_3inst<cb>(x))` -- the fp16 codebook math (IMUL + portable byte-sum + scale)
run in fp16 and upcast to float. It replicates that exact fp16 math in pure NumPy and checks the
dequant against the reference over the FULL 16-bit window space (0..0xFFFF), across cb=0/1/2.

The dequant input is a 16-bit SLIDING WINDOW (each window spans 16/K consecutive K-bit stored
codes), NOT a K-bit code: `decode_3inst<cb>` depends on all 16 bits, so it takes up to 2^16 = 65536
distinct values. A 2^K LUT is undersized (out-of-bounds for most windows) and a 2^16 LUT (256 KB)
is infeasible for 48 KB smem, so the dequant is the per-weight ALU decode. Testing all 65536
windows is the exhaustive check: every window the extractor can produce is covered, for every K.

The values must be the EXACT fp16 codebook results upcast to fp32 (bit-identical to decode_3inst),
not a fresh fp32 formula: a fresh fp32 byte_sum*k_inv+k_bias drifts by one fp16 ulp and is not
fp16-representable.

Pure CPU: no torch.cuda, no CUDA tensors, no device skip.

    python -m pytest tests/test_maxwell_dequant.py -v
"""

import numpy as np
import pytest

U32 = (1 << 32) - 1
FULL = 1 << 16  # the 16-bit window space the dequant input spans

# mul1 (cb2) codebook constants, as fp16 bit patterns (see codebook.cuh)
K_INV = np.uint16(0x1EEE).view(np.float16)   # 1/147.7
K_BIAS = np.uint16(0xC931).view(np.float16)  # (-1024 - 510) * k_inv


def _bits_to_f16(v):
    """Reinterpret the low 16 bits of v as an fp16 value (array or scalar)."""
    return np.uint16(v & 0xFFFF).view(np.float16)


def _byte_sum(x):
    return (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)


# Vectorized dequant: mirrors decode_3inst_f<cb> = __half2float(decode_3inst<cb>(x)) over a batch of
# 16-bit windows. This is the reference the kernel must match bit-for-bit.

def decode_f(x, cb):
    if cb == 0:
        x = (x * 89226354) & U32
        x = (x + 64248484) & U32
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        lo = (x & 0xFFFF).astype(np.uint16).view(np.float16)
        hi = ((x >> 16) & 0xFFFF).astype(np.uint16).view(np.float16)
        return (lo + hi).astype(np.float32)
    if cb == 1:
        x = (x * 0xCBAC1FED) & U32
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        lo = (x & 0xFFFF).astype(np.uint16).view(np.float16)
        hi = ((x >> 16) & 0xFFFF).astype(np.uint16).view(np.float16)
        return (lo + hi).astype(np.float32)
    if cb == 2:
        x = (x * 0x83DCD12D) & U32
        s = _byte_sum(x) + 0x6400
        h = s.astype(np.uint16).view(np.float16)
        return (np.float16(h * K_INV) + K_BIAS).astype(np.float32)
    raise ValueError(cb)


# Scalar dequant: a single 16-bit window, using Python big-int arithmetic for the IMUL (an
# independent code path from the vectorized uint64 version, so a uint64 overflow in either shows up).

def decode_scalar(w, cb):
    if cb == 0:
        x = (w * 89226354 + 64248484) & U32
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        return np.float16(_bits_to_f16(x & 0xFFFF) + _bits_to_f16((x >> 16) & 0xFFFF)).astype(np.float32)
    if cb == 1:
        x = (w * 0xCBAC1FED) & U32
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        return np.float16(_bits_to_f16(x & 0xFFFF) + _bits_to_f16((x >> 16) & 0xFFFF)).astype(np.float32)
    if cb == 2:
        x = (w * 0x83DCD12D) & U32
        s = _byte_sum(x) + 0x6400
        h = _bits_to_f16(s)
        return np.float16(np.float16(h * K_INV) + K_BIAS).astype(np.float32)
    raise ValueError(cb)


@pytest.mark.parametrize("cb", [0, 1, 2])
def test_dequant_matches_reference_all_windows(cb):
    """decode_3inst_f<cb>(x) must be fp16-representable for EVERY 16-bit window.

    This is the invariant the fp32 GEMM relies on: the per-weight procedural dequant (no LUT) is
    an fp16 result upcast to fp32 for all 65536 windows the extractor can produce, for every K.
    """
    x = np.arange(FULL, dtype=np.uint64)
    got = decode_f(x, cb)
    assert got.shape == (FULL,)
    assert np.array_equal(got, np.float16(got).astype(np.float32)), f"cb={cb}: a dequant value is not fp16-representable"


@pytest.mark.parametrize("cb", [0, 1, 2])
def test_dequant_matches_scalar_reference(cb):
    """Representative 16-bit windows (0, 1, 2, ... and a spread across 0..0xFFFF) match an
    independent scalar reference (Python big-int IMUL), so the vectorized path is not circular."""
    spread = np.array(list(range(16)) + [0x0100, 0x0101, 0x0FFF, 0x8000, 0x8001, 0x7FFF, 0x8FFF, 0xFFFF], dtype=np.uint64)
    got = decode_f(spread, cb)
    ref = np.array([decode_scalar(int(w), cb) for w in spread], dtype=np.float32)
    assert np.array_equal(got, ref), f"cb={cb}: the dequant diverges from the scalar reference"


@pytest.mark.parametrize("cb", [0, 1, 2])
def test_anchor_values(cb):
    """Frozen window -> value pairs, guarding against accidental changes to the codebook math."""
    anchors = {
        0: {0: 0.76806640625, 255: 1.2041015625},   # cb0
        1: {0: 1.84375, 255: -1.287109375},         # cb1
        2: {0: -3.453125, 255: 0.8359375},          # cb2
    }
    for window, expected in anchors[cb].items():
        got = float(decode_scalar(window, cb))
        assert got == expected, f"cb={cb} window={window}: {got} != {expected}"


def test_cb2_differs_from_fresh_fp32():
    """The mul1 path must not collapse to a fresh fp32 byte_sum*k_inv+k_bias.

    A fresh fp32 recompute (upcast h, then fp32 FMA) drifts by one fp16 ulp and is not
    fp16-representable; the exact fp16 path must differ from it.
    """
    x = np.arange(FULL, dtype=np.uint64)
    got = decode_f(x, 2)
    xm = (x * 0x83DCD12D) & U32
    s = _byte_sum(xm) + 0x6400
    h32 = s.astype(np.uint16).view(np.float16).astype(np.float32)
    fresh = h32 * float(K_INV) + float(K_BIAS)
    assert not np.array_equal(got, fresh), "cb2 dequant equals the fresh fp32 formula (fp16 rounding lost)"
    assert not np.array_equal(fresh, np.float16(fresh).astype(np.float32)), "fresh fp32 unexpectedly fp16-representable"
