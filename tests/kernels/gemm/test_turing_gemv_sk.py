"""
sm_75 split-k GEMV (exllamav3_ext/quant/exl3_gemv_sk_kernel.cuh), the path ext.exl3_gemm takes on Turing for
1 <= m <= 16: 2-4 bpw at n <= 12288 (above that the wide block-per-group config stays a little faster), and 5-8 bpw
(typically the output head) at any n, including n large enough that one warp's range spans several column groups.

Every case is checked against reconstruct-then-matmul (ext.reconstruct_had_slice, the independently decoded weight)
and against the block-per-group GEMV the split-k path replaces (EXL3_GEMV_SK=0, read per call by the extension), for
every bitrate and codebook instance and for fp16 and fp32 outputs.

Tolerance: the kernel accumulates 8-term runs of the MMA in fp16 before folding to fp32, so its error is a few fp16
unit roundoffs relative to the output scale. Max error below 4e-3 of the output's max, and within 2x + 1e-3 of the
block-per-group GEMV's error (the same weight is decoded by both). A tile mapped or decoded wrongly is an O(1)
error. Repeat launches must be bit-identical.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import generator, rand_scale, rand_trellis

# Exactly Turing: the split-k kernel traps on every other target, and the wide-config and 5-8 bpw boundaries the
# cases probe are sm_75 rules
pytestmark = [pytest.mark.cuda_only, pytest.mark.cc(7, 5), pytest.mark.cc_max(7, 5)]


def _weights(k, n, K, seed, device):
    g = generator(seed)
    return {
        "trellis": rand_trellis(k, n, K, g, device),
        "suh": rand_scale(k, g, device),
        "svh": rand_scale(n, g, device),
    }


def _gemm(A, w, cb, dtype, device, sk, monkeypatch):
    monkeypatch.setenv("EXL3_GEMV_SK", "1" if sk else "0")
    C = torch.empty((A.shape[0], w["svh"].numel()), dtype = dtype, device = device)
    ext.exl3_gemm(A, w["trellis"], C, w["suh"], torch.empty_like(A), w["svh"], -1, cb == "mcg", cb == "mul1", -1)
    torch.cuda.synchronize(device)
    return C


def _reference(A, w, K, cb, device):
    k, n = w["trellis"].shape[0] * 16, w["trellis"].shape[1] * 16
    B = torch.empty((k, n), dtype = torch.half, device = device)
    ext.reconstruct_had_slice(B, w["trellis"], w["suh"], w["svh"], K, cb == "mcg", cb == "mul1", 0)
    return A.float() @ B.float()


def _check(err_sk, err_old, K, cb, m, k, n):
    tag = f"K = {K} {cb}, m = {m}, k = {k}, n = {n}"
    assert err_sk < 4e-3, f"{tag}: split-k GEMV vs reference {err_sk:.2e} (old GEMV {err_old:.2e})"
    assert err_sk < 2 * err_old + 1e-3, f"{tag}: split-k GEMV {err_sk:.2e} well above the old GEMV's {err_old:.2e}"


@pytest.mark.parametrize("K,cb", [(4, "3inst"), (4, "mcg"), (4, "mul1"), (3, "mcg"), (3, "mul1"), (2, "mcg"), (2, "mul1")])
@pytest.mark.parametrize("m", [1, 2, 5, 8, 12, 16])
@pytest.mark.parametrize("k,n", [(5120, 6144), (6144, 5120), (2048, 1024), (1024, 12288)])
@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@torch.inference_mode()
def test_gemv_sk_matches_reference(device, monkeypatch, K, cb, m, k, n, dtype):
    g = generator(k * 31 + n + m * 7 + K)
    A = (torch.randn(m, k, generator = g) * 0.5).half().to(device)
    w = _weights(k, n, K, k * 31 + n + m * 7 + K, device)
    ref = _reference(A, w, K, cb, device)
    scale = ref.abs().max().item()
    y_sk = _gemm(A, w, cb, dtype, device, True, monkeypatch).float()
    y_old = _gemm(A, w, cb, dtype, device, False, monkeypatch).float()
    _check((y_sk - ref).abs().max().item() / scale, (y_old - ref).abs().max().item() / scale, K, cb, m, k, n)


@pytest.mark.parametrize("K,cb", [(5, "mul1"), (6, "3inst"), (6, "mcg"), (6, "mul1"), (8, "mcg")])
@pytest.mark.parametrize("m", [1, 3, 8])
# n = 131072 gives 2048 column groups, more than there are warps: a warp's range then covers whole groups, which go
# straight to the workspace instead of the block's shared partials
@pytest.mark.parametrize("k,n", [(5120, 4096), (1024, 131072)])
@torch.inference_mode()
def test_gemv_sk_high_bitrate(device, monkeypatch, K, cb, m, k, n):
    g = generator(k + n + m * 7 + K)
    A = (torch.randn(m, k, generator = g) * 0.5).half().to(device)
    w = _weights(k, n, K, k + n + m * 7 + K, device)
    ref = _reference(A, w, K, cb, device)
    scale = ref.abs().max().item()
    y_sk = _gemm(A, w, cb, torch.half, device, True, monkeypatch).float()
    y_old = _gemm(A, w, cb, torch.half, device, False, monkeypatch).float()
    _check((y_sk - ref).abs().max().item() / scale, (y_old - ref).abs().max().item() / scale, K, cb, m, k, n)


@pytest.mark.parametrize("m", [1, 8])
@torch.inference_mode()
def test_gemv_sk_deterministic(device, monkeypatch, m):
    k, n = 8704, 5120
    g = generator(5)
    A = (torch.randn(m, k, generator = g) * 0.5).half().to(device)
    w = _weights(k, n, 4, 5, device)
    ys = [_gemm(A, w, "mul1", torch.float, device, True, monkeypatch) for _ in range(4)]
    for y in ys[1:]:
        assert torch.equal(y, ys[0])
