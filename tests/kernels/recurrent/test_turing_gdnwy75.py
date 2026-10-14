"""
gdnwy75 (exllamav3_ext/turing/gdnwy75.cu): the gated delta rule WY representation (FLA's fused kkt + solve_tril
+ recompute_w_u) on sm_75 HMMA (mma.sync.m16n8k8), against FLA's Triton kernels, at Qwen3.8-27B GDN shapes
(16 key heads, 48 value heads, K = V = 128). The kernel is plain PTX for sm_75 and later, so this runs on every
CUDA device.
"""

import pytest
import torch

from exllamav3.vendor.fla.gdn_chunk_fwd import chunk_gated_delta_rule_fwd_intra
from exllamav3.vendor.fla.cumsum import chunk_local_cumsum
from exllamav3.vendor.fla.l2norm import l2norm_fwd

pytestmark = [pytest.mark.cuda_only, pytest.mark.cc(7, 5)]

BT = 64


def _rel(a, b):
    den = b.float().abs().max().item()
    return (a.float() - b.float()).abs().max().item() / (den if den > 0 else 1.0)


@pytest.mark.parametrize("B, T", [(1, 2048), (1, 1000), (1, 37), (2, 300), (1, 64)])
def test_gdnwy75_matches_triton(monkeypatch, device, B, T):
    gen = torch.Generator(device = device).manual_seed(T + B)
    H, HV = 16, 48
    k = l2norm_fwd(torch.randn(B, T, H, 128, device = device, dtype = torch.half, generator = gen))[0]
    v = torch.randn(B, T, HV, 128, device = device, dtype = torch.half, generator = gen)
    beta = torch.rand(B, T, HV, device = device, generator = gen).half()
    g = chunk_local_cumsum(-torch.rand(B, T, HV, device = device, generator = gen) * 0.1, chunk_size = BT, scale = 1.4426950216)

    monkeypatch.setenv("EXL3_GDN_WY_CUDA", "0")
    w_ref, u_ref, _ = chunk_gated_delta_rule_fwd_intra(k = k, v = v, g = g, beta = beta, chunk_size = BT)
    monkeypatch.setenv("EXL3_GDN_WY_CUDA", "1")
    w, u, a = chunk_gated_delta_rule_fwd_intra(k = k, v = v, g = g, beta = beta, chunk_size = BT)
    torch.cuda.synchronize()

    assert a is None
    # fp16 storage of the solved matrix and of w / u bounds the agreement (measured ~7-9e-4)
    assert _rel(w, w_ref) < 3e-3
    assert _rel(u, u_ref) < 3e-3
