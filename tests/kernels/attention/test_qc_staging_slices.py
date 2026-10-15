"""
Bounded KV-sliced staging of the quantized-cache prefill (triton_paged._qc_staging_pages): a window wider
than the fixed fp16 scratch is dequantized and attended one block-table slice at a time into the same buffer,
with the running softmax state merged across slices, so the result must match the single-launch path over the
same inputs. The small-bound and large-bound runs use the same fp16 kernel (EXL3_SDPA_PREFILL is forced off,
which also lowers the two-pass query-length threshold to 256), isolating the slicing arithmetic from the sm_75
SDPA / fa75 path, where SDPA exposes no lse and a sliced window cannot be merged.
"""
import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.attention_fn import triton_paged
from exllamav3.modules.attention_fn.triton_paged import (
    _paged_kv_update, paged_attn_triton_prefill,
)
from testlib.attention import gather_pages, quant_roundtrip, rand_paged_cache, ref_attn
from testlib.compare import assert_rel_close


def run_staged(q, qk, qv, sk, sv, bt, sl, kvh, bits, q_len, device, bound):
    """One two-pass prefill with the staging window bound to `bound` cache pages"""
    triton_paged._qc_staging_pages = bound
    return paged_attn_triton_prefill(
        q, None, None, qk, qv, bt, sl, causal = True,
        qc = (sk, sv, bits, bits), pre_appended_len = q_len, n_kv_heads_override = kvh,
    )


# (bsz, past, q_len, head_dim); the 2324-context rows are the case where query rows span a slice boundary
CASES = [
    (1, 1800, 300, 128),
    (3, 1800, 300, 128),
    (1, 1800, 300, 256),
    (3, 1800, 300, 256),
    (1, 1800, 2048, 128),
    (3, 1800, 2048, 128),
    (1, 1800, 2048, 256),
    (3, 1800, 2048, 256),
    (1, 2324, 300, 128),
    (3, 2324, 300, 128),
    (1, 2324, 300, 256),
]


@pytest.mark.parametrize("bsz,past,q_len,hd",
                         [pytest.param(*c, id = "B{}-T{}-Q{}-D{}".format(*c)) for c in CASES])
def test_qc_staging_slices(device, monkeypatch, bsz, past, q_len, hd):
    monkeypatch.setenv("EXL3_SDPA_PREFILL", "0")   # same Triton fp16 kernel on both sides
    monkeypatch.setattr(triton_paged, "_qc_staging", 1)
    torch.manual_seed(bsz * 1000 + past + q_len + hd)
    kvh, n_heads, bits = 2, 8, 4
    T = past + q_len

    kc, vc, bt, sl = rand_paged_cache(bsz, T + 3, kvh, hd, past, device)
    k = torch.randn((bsz, q_len, kvh, hd), dtype = torch.half, device = device)
    v = torch.randn_like(k)
    q = torch.randn((bsz, q_len, n_heads, hd), dtype = torch.half, device = device)
    _paged_kv_update(k, v, kc, vc, bt, sl, bt.shape[1], bsz, q_len, kvh, PAGE_SIZE, hd)

    pages = kc.shape[0]
    qk, sk, kdeq = quant_roundtrip(kc.view(pages, PAGE_SIZE, kvh * hd), bits)
    qv, sv, vdeq = quant_roundtrip(vc.view(pages, PAGE_SIZE, kvh * hd), bits)
    kdeq = kdeq.view(pages, PAGE_SIZE, kvh, hd)
    vdeq = vdeq.view(pages, PAGE_SIZE, kvh, hd)

    # Two cache pages per slice: every case's window spans several slices, and 2324 + 300 straddles one
    out_small = run_staged(q, qk, qv, sk, sv, bt, sl, kvh, bits, q_len, device, 2)
    out_large = run_staged(q, qk, qv, sk, sv, bt, sl, kvh, bits, q_len, device, 1 << 20)

    assert_rel_close(out_small, out_large, 2e-3, "sliced vs single-slice")
    ref = ref_attn(q, gather_pages(kdeq, bt, T), gather_pages(vdeq, bt, T), causal = True)
    assert_rel_close(out_small, ref, 1.2e-2, "sliced vs fp32 reference")
