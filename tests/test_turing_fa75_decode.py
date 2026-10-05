"""
fa75_decode_qc (sm_75 CUDA-core decode attention for the packed 4-bit cache, head_dim 256, GQA 8)
against an fp32 torch reference and the Triton quantized-cache decode oracle. Split-K counts 1..64.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.modules.attention_fn.triton_paged import (
    paged_attn_triton_decode, paged_attn_fa75_decode_qc,
)
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.constants import PAGE_SIZE

device = "cuda:0"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (7, 5),
    reason = "needs an sm_75+ CUDA device",
)


def gather(kc, bt, T):
    B, pages = bt.shape
    flat = kc[bt.long().view(-1)].view(B, pages * PAGE_SIZE, kc.shape[2], kc.shape[3])
    return flat[:, :T]


def ref_attn(q, k, v):
    """q (B,1,H,D) attends to all T keys of k/v (B,T,KVH,D); GQA head h -> kv head h//(H//KVH).
    Loops per q head so the fp32 reference stays small at long context."""
    B, Q, H, D = q.shape
    T, KVH = k.shape[1], k.shape[2]
    g = H // KVH
    out = torch.empty_like(q)
    for h in range(H):
        kh = h // g
        kf = k[:, :, kh, :].float()
        vf = v[:, :, kh, :].float()
        s = torch.einsum("bqd,bkd->bqk", q[:, :, h, :].float(), kf) * D ** -0.5
        out[:, :, h, :] = torch.einsum("bqk,bkd->bqd", torch.softmax(s, -1), vf).half()
    return out


def _quant_cache(kc, bits):
    pages, ps, kvh, hd = kc.shape
    rows = pages * ps
    pq = torch.empty((rows, kvh * hd // 32 * bits), dtype = torch.int32, device = device)
    sc = torch.empty((rows, kvh * hd // 32), dtype = torch.half, device = device)
    ext.quant_cache_cont(kc.reshape(rows, kvh * hd).contiguous(), pq, sc, 0.0)
    deq = torch.empty((rows, kvh * hd), dtype = torch.half, device = device)
    ext.dequant_cache_cont(pq, sc, deq, 0.0)
    return pq.view(pages, ps, -1), sc.view(pages, ps, -1), deq.view(pages, ps, kvh, hd)


def make_case(bsz, past, seed, kvh = 2, hd = 256):
    torch.manual_seed(seed)
    T = past + 1
    pages = -(-T // PAGE_SIZE)
    kc = torch.randn((bsz * pages, PAGE_SIZE, kvh, hd), dtype = torch.half, device = device)
    vc = torch.randn_like(kc)
    perm = torch.randperm(bsz * pages, device = device, dtype = torch.int32)
    bt = perm.view(bsz, pages)
    sl = torch.full((bsz,), past, dtype = torch.int32, device = device)
    q = torch.randn((bsz, 1, kvh * 8, hd), dtype = torch.half, device = device)
    return kc, vc, bt, sl, q, T


@pytest.mark.parametrize("past", [700, 4096, 40000])
@pytest.mark.parametrize("bsz", [1, 2])
@pytest.mark.parametrize("splits", [1, 3, 16, 64])
def test_fa75_decode_qc(bsz, past, splits):
    torch.cuda.set_device(device)
    kvh, hd, bits = 2, 256, 4
    kc, vc, bt, sl, q, T = make_case(bsz, past, seed = past * 13 + bsz)
    qk, sk, kdeq = _quant_cache(kc, bits)
    qv, sv, vdeq = _quant_cache(vc, bits)

    out = paged_attn_fa75_decode_qc(
        q = q, k_cache = qk, k_scales = sk, v_cache = qv, v_scales = sv,
        block_table = bt, cache_seqlens = sl,
        n_q_heads = kvh * 8, n_kv_heads = kvh, max_kv_len = None,
        softmax_scale = hd ** -0.5, num_splits = splits,
    )

    ref = ref_attn(q, gather(kdeq, bt, T), gather(vdeq, bt, T))
    err_ref = (out.float() - ref).abs().max().item() / ref.abs().max().item()

    oracle = paged_attn_triton_decode(
        q, None, None, qk, qv, bt, sl, causal = True, qc = (sk, sv, bits, bits),
        pre_appended_len = 1, n_kv_heads_override = kvh, num_splits = splits,
    )
    err_or = (out.float() - oracle.float()).abs().max().item() / oracle.float().abs().max().item()

    assert torch.isfinite(out).all(), "non-finite output"
    assert err_or < 2e-2, f"vs triton oracle rel err {err_or:.3e} (ref {err_ref:.3e})"
    assert err_ref < 3e-2, f"vs fp32 ref rel err {err_ref:.3e}"
