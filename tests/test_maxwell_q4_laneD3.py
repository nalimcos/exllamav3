import pytest
import torch

from test_maxwell_q4_decode import (
    DEV, check_close, make_case, make_score_case, reference, reference_rotated,
)
from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (5, 2),
    reason="requires Maxwell sm_52",
)


def run_laneD3(case, splits=30):
    q, kc, ks, vc, vs, bt, past, out, qr, po, ml = case
    ext.q4m_laneD3_attention(q, kc, ks, vc, vs, bt, past, out, qr, po, ml, 1 / 16, splits)
    return out


def check_case(case, out, label):
    q, kc, ks, vc, vs, bt, past = case[:7]
    ref = reference(q, kc, ks, vc, vs, bt, int(past.item()), 1 / 16)
    check_close(out, ref, label)


@pytest.mark.parametrize("total", [7, 8, 9, 15, 16, 17])
def test_laneD3_partial_eight_token_batches(total):
    case = make_case(total, seed=7100 + total, splits=30)
    out = run_laneD3(case)
    check_case(case, out, f"laneD3 partial batch {total}")


@pytest.mark.parametrize("total", [1, 33, 65, 129, 257])
def test_laneD3_empty_warps_and_finalization_determinism(total):
    case = make_case(total, seed=7200 + total, splits=1)
    out = run_laneD3(case, splits=1)
    check_case(case, out, f"laneD3 finalization {total}")
    expected = out.clone()
    for _ in range(16):
        out = run_laneD3(case, splits=1)
        assert torch.equal(out, expected), f"nondeterministic finalization at {total}"
        assert torch.isfinite(out).all()


def test_laneD3_asynchronous_row_thresholds_and_replay():
    scores = torch.zeros(1024, dtype=torch.float32, device=DEV)
    scores[128:] = 2.1
    scores[640:] = 4.0
    case = make_score_case(scores, splits=30, seed=7300)
    q = case[0]
    q.zero_()
    for r in range(6):
        q[:, :, r::6, :].fill_(float(r + 1))
    out = run_laneD3(case)
    q, kc, ks, vc, vs, bt, past = case[:7]
    ref = reference_rotated(q, kc, ks, vc, vs, bt, int(past.item()), 1 / 16)
    check_close(out, ref, "laneD3 asynchronous row thresholds")
    expected = out.clone()
    for _ in range(8):
        out = run_laneD3(case)
        assert torch.equal(out, expected)


def test_laneD3_resource_failure_selects_generic(monkeypatch):
    from exllamav3.modules.attention_fn.common import AttnArgs
    from exllamav3.modules.attention_fn.maxwell_q4 import fn_maxwell_q4_decode

    case = make_case(17, seed=7400, splits=30)
    q, kc, ks, vc, vs, bt, past = case[:7]
    args = AttnArgs(
        bsz=1, q_len=1, num_q_heads=24, dim=256, kv_len=1, num_kv_heads=4,
        q=q, k=q, v=q, k_cache=None, v_cache=None, causal=True, sm_scale=1 / 16,
        cu_seqlens=None, max_seqlen=None, window_size=None, softcap=0.0,
        block_table=bt, cache_seqlens=past, q_cache=(kc, ks, vc, vs, 4, 4),
    )
    monkeypatch.setenv("EXL3_ATTN_Q4M", "3")
    out = fn_maxwell_q4_decode(args)
    resources = ext.q4m_laneD3_resource_info()
    resource_failure = resources[2] > 96 or resources[5] < 5
    if resource_failure:
        assert out is None
    else:
        check_case(case, out, "laneD3 eager dispatch")
    monkeypatch.setenv("EXL3_ATTN_Q4M", "unsupported")
    assert fn_maxwell_q4_decode(args) is None


@pytest.mark.parametrize("total", [7, 9, 32, 33, 128, 129, 256, 257])
def test_laneD3_sanitizer_native(total):
    case = make_case(total, seed=7500 + total, splits=30)
    out = run_laneD3(case)
    assert torch.isfinite(out).all()
