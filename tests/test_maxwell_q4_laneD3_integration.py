import pytest
import torch

from test_maxwell_q4_decode import DEV, check_close, reference
from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (5, 2),
    reason="requires Maxwell sm_52",
)


def workspace_rows(bsz, q_len):
    rows = bsz * 4 * (6 if q_len == 1 else (q_len // 2) * 12)
    if q_len > 1 and q_len & 1:
        rows += bsz * 4 * 6
    return rows


def make_case(bsz, q_len, past, splits=30, seed=0):
    torch.manual_seed(seed)
    past = torch.tensor(past, dtype=torch.int32, device=DEV)
    pages = max((int(x) + q_len + 255) // 256 for x in past)
    storage_pages = bsz * pages + 3
    bt = torch.randperm(storage_pages, device=DEV, dtype=torch.int32)[:bsz * pages].view(bsz, pages)
    kc = torch.randint(-(2**31), 2**31 - 1, (storage_pages, 256, 64),
                       dtype=torch.int64, device=DEV).view(torch.int32)
    vc = torch.randint(-(2**31), 2**31 - 1, (storage_pages, 256, 64),
                       dtype=torch.int64, device=DEV).view(torch.int32)
    ks = torch.rand((storage_pages, 256, 32), dtype=torch.float16, device=DEV) * 0.2 + 0.02
    vs = torch.rand((storage_pages, 256, 32), dtype=torch.float16, device=DEV) * 0.2 + 0.02
    q = torch.randn((bsz, q_len, 24, 256), dtype=torch.float16, device=DEV)
    out = torch.empty_like(q)
    qr = torch.empty(q.shape, dtype=torch.float32, device=DEV)
    rows = workspace_rows(bsz, q_len)
    po = torch.full((rows * splits * 256,), float("nan"), dtype=torch.float32, device=DEV)
    ml = torch.full((rows * splits * 2,), float("nan"), dtype=torch.float32, device=DEV)
    return q, kc, ks, vc, vs, bt.contiguous(), past, out, qr, po, ml


def run_case(case, splits=30):
    q, kc, ks, vc, vs, bt, past, out, qr, po, ml = case
    ext.q4m_laneD3_attention(q, kc, ks, vc, vs, bt, past, out, qr, po, ml, 1 / 16, splits)
    return out


def reference_case(case):
    q, kc, ks, vc, vs, bt, past = case[:7]
    return torch.cat([
        reference(q[b:b + 1], kc, ks, vc, vs, bt[b:b + 1], int(past[b]), 1 / 16)
        for b in range(q.shape[0])
    ])


def check_case(case, out, label):
    check_close(out, reference_case(case), label)


@pytest.mark.parametrize("bsz", [1, 2, 3, 4])
@pytest.mark.parametrize("q_len", list(range(1, 9)))
@pytest.mark.parametrize("splits", [1, 30])
def test_batch_query_matrix(bsz, q_len, splits):
    bases = [0, 31, 127, 253]
    past = [bases[(b + q_len) % 4] + 3 * b for b in range(bsz)]
    case = make_case(bsz, q_len, past, splits, seed=81000 + 1000 * bsz + 10 * q_len + splits)
    out = run_case(case, splits)
    check_case(case, out, f"batch={bsz}, q_len={q_len}, splits={splits}")


def test_even_row_excludes_odd_future_token():
    case = list(make_case(2, 2, [31, 63], 30, seed=82000))
    first = run_case(case).clone()
    q, kc, ks, vc, vs, bt, past = case[:7]
    for b in range(2):
        logical = int(past[b]) + 1
        row = int(bt[b, logical // 256]) * 256 + logical % 256
        kc.view(-1, 128)[row].bitwise_xor_(0x55555555)
        vc.view(-1, 128)[row].bitwise_xor_(0x33333333)
    second = run_case(case)
    assert torch.equal(second[:, 0], first[:, 0])
    check_case(case, second, "odd-only future-token poisoning")


def test_default_eager_and_explicit_fallback(monkeypatch):
    from exllamav3.modules.attention_fn.common import AttnArgs
    from exllamav3.modules.attention_fn.maxwell_q4 import fn_maxwell_q4_decode

    case = make_case(2, 3, [31, 63], 30, seed=83000)
    q, kc, ks, vc, vs, bt, past = case[:7]
    args = AttnArgs(
        bsz=2, q_len=3, num_q_heads=24, dim=256, kv_len=3, num_kv_heads=4,
        q=q, k=q, v=q, k_cache=None, v_cache=None, causal=True, sm_scale=1 / 16,
        cu_seqlens=None, max_seqlen=None, window_size=None, softcap=0.0,
        block_table=bt, cache_seqlens=past, q_cache=(kc, ks, vc, vs, 4, 4),
    )
    monkeypatch.delenv("EXL3_ATTN_Q4M", raising=False)
    out = fn_maxwell_q4_decode(args)
    assert out is not None
    check_case(case, out, "default eager batch/query")
    monkeypatch.setenv("EXL3_ATTN_Q4M", "0")
    assert fn_maxwell_q4_decode(args) is None


def test_capture_replay_long_short_and_page_permutation():
    case = list(make_case(2, 3, [126, 254], 30, seed=84000))
    expected = run_case(case).clone()
    check_case(case, expected, "pre-capture")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_case(case)
    case[7].fill_(float("nan"))
    graph.replay()
    assert torch.equal(case[7], expected)

    case[6].copy_(torch.tensor([1, 7], dtype=torch.int32, device=DEV))
    case[0].normal_()
    case[8].fill_(float("nan"))
    case[9].fill_(float("nan"))
    graph.replay()
    check_case(case, case[7], "long-to-short replay")

    storage_pages = case[1].shape[0]
    pages = case[5].shape[1]
    replacement = torch.randperm(storage_pages, device=DEV, dtype=torch.int32)[:2 * pages].view(2, pages)
    case[5].copy_(replacement)
    case[6].copy_(torch.tensor([257, 129], dtype=torch.int32, device=DEV))
    graph.replay()
    check_case(case, case[7], "permuted-page replay")


@pytest.mark.parametrize("bsz,q_len", [(2, 3), (4, 8)])
def test_sanitizer_batch_query(bsz, q_len):
    case = make_case(bsz, q_len, [31 + 17 * b for b in range(bsz)], 30,
                     seed=85000 + 10 * bsz + q_len)
    out = run_case(case)
    assert torch.isfinite(out).all()
