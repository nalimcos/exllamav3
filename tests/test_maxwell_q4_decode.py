import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (5, 2),
    reason="requires Maxwell sm_52",
)
DEV = torch.device("cuda:0")
R32 = 1.0 / math.sqrt(32.0)
LOG2E = math.log2(math.e)


def h32_matrix():
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    while h.shape[0] < 32:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h.float().to(DEV)


def inverse_rotate(x):
    # x: [..., 256], stored in the normalized rotated domain
    return (x.reshape(*x.shape[:-1], 8, 32) @ h32_matrix()).reshape(x.shape) * R32


def unpack_cache(words, scales, block_table, length):
    flat_words = words.reshape(-1, 128)
    flat_scales = scales.reshape(-1, 32)
    logical = torch.arange(length, device=DEV, dtype=torch.int32)
    pages = block_table.reshape(-1)[logical // 256].long()
    rows = pages * 256 + (logical % 256).long()
    w = flat_words[rows].view(length, 4, 8, 4).to(torch.int64) & 0xFFFFFFFF
    nibbles = (w[..., None] >> (4 * torch.arange(8, device=DEV))) & 15
    s = flat_scales[rows].view(length, 4, 8, 1, 1).float()
    rotated = (nibbles.float() - 7.5) * (s * 0.125)
    return inverse_rotate(rotated.view(length, 4, 256))


def reference(q, kc, ks, vc, vs, block_table, past, scale):
    total = past + q.shape[1]
    k = unpack_cache(kc, ks, block_table[0], total)
    v = unpack_cache(vc, vs, block_table[0], total)
    k = k.repeat_interleave(6, dim=1)
    v = v.repeat_interleave(6, dim=1)
    outputs = []
    for qpos in range(q.shape[1]):
        scores = torch.einsum("bhd,nhd->bhn", q[:, qpos].float(), k) * scale
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("bhn,nhd->bhd", probs, v))
    return torch.stack(outputs, dim=1)


def make_case(total, seed=0, splits=30, scale_pattern=None):
    torch.manual_seed(seed)
    pages = (total + 255) // 256
    # Include unused storage and a noncontiguous physical-page permutation.
    storage_pages = pages + 3
    permutation = torch.randperm(storage_pages, device=DEV, dtype=torch.int32)
    block_table = permutation[:pages].view(1, pages).contiguous()
    kc = torch.randint(-(2**31), 2**31 - 1, (storage_pages, 256, 64),
                       dtype=torch.int64, device=DEV).view(torch.int32)
    vc = torch.randint(-(2**31), 2**31 - 1, (storage_pages, 256, 64),
                       dtype=torch.int64, device=DEV).view(torch.int32)
    if scale_pattern is None:
        ks = torch.rand((storage_pages, 256, 32), dtype=torch.float16, device=DEV) * 0.2 + 0.02
        vs = torch.rand((storage_pages, 256, 32), dtype=torch.float16, device=DEV) * 0.2 + 0.02
    else:
        ks = scale_pattern(storage_pages).half()
        vs = torch.rand((storage_pages, 256, 32), dtype=torch.float16, device=DEV) * 0.2 + 0.02
    q = torch.randn((1, 1, 24, 256), dtype=torch.float16, device=DEV)
    past = torch.tensor([total - 1], dtype=torch.int32, device=DEV)
    out = torch.empty_like(q)
    q_rot = torch.empty(q.shape, dtype=torch.float32, device=DEV)
    po = torch.full((4 * splits * 6 * 256,), float("nan"), dtype=torch.float32, device=DEV)
    ml = torch.full((4 * splits * 6 * 2,), float("nan"), dtype=torch.float32, device=DEV)
    return q, kc, ks, vc, vs, block_table, past, out, q_rot, po, ml


def check_close(out, ref, label=""):
    assert torch.isfinite(out).all(), label
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=0, msg=label)
    delta = (out.float() - ref).square().mean().sqrt()
    denominator = ref.square().mean().sqrt().clamp_min(1e-6)
    rel = (delta / denominator).item()
    per_head = ((out.float() - ref).square().mean(dim=(0, 1, 3)).sqrt() /
                ref.square().mean(dim=(0, 1, 3)).sqrt().clamp_min(1e-6))
    assert rel <= 1e-3, f"{label}: rel_rms={rel}"
    assert per_head.max().item() <= 1e-3, f"{label}: per-head rel_rms={per_head}"


def run_native(case, splits=30, scale=1 / 16):
    q, kc, ks, vc, vs, bt, past, out, qr, po, ml = case
    ext.q4m_attention(q, kc, ks, vc, vs, bt, past, out, qr, po, ml, scale, splits)
    return out


def test_prepare_h32():
    torch.manual_seed(4)
    q = torch.randn((1, 1, 24, 256), dtype=torch.float16, device=DEV)
    qr = torch.empty(q.shape, dtype=torch.float32, device=DEV)
    scale = 0.731
    ext.q4m_prepare(q, qr, scale)
    ref = (q.float().view(1, 1, 24, 8, 32) @ h32_matrix()).view(q.shape) * (scale * R32)
    torch.testing.assert_close(qr, ref, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("total", [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 258])
@pytest.mark.parametrize("splits", [1, 30])
def test_decode_boundaries(total, splits):
    case = make_case(total, seed=total * 37 + splits, splits=splits)
    out = run_native(case, splits=splits)
    q, kc, ks, vc, vs, bt, past = case[:7]
    ref = reference(q, kc, ks, vc, vs, bt, int(past.item()), 1 / 16)
    check_close(out, ref, f"total={total}, splits={splits}")


def test_long_to_short_reuses_poisoned_partials():
    case = make_case(8326, seed=991, splits=30)
    out = run_native(case)
    q, kc, ks, vc, vs, bt, past = case[:7]
    ref = reference(q, kc, ks, vc, vs, bt, int(past.item()), 1 / 16)
    check_close(out, ref, "8326")
    short = list(make_case(1, seed=992, splits=30))
    # Reuse the long partial allocations, now containing valid old numerators and metadata.
    short[8].fill_(float("nan"))
    short[9], short[10] = case[9], case[10]
    out = run_native(short)
    q, kc, ks, vc, vs, bt, past = short[:7]
    ref = reference(q, kc, ks, vc, vs, bt, int(past.item()), 1 / 16)
    check_close(out, ref, "long-to-short")


def unpack_rotated(words, scales, block_table, length):
    flat_words = words.reshape(-1, 128)
    flat_scales = scales.reshape(-1, 32)
    logical = torch.arange(length, device=DEV, dtype=torch.int32)
    pages = block_table.reshape(-1)[logical // 256].long()
    rows = pages * 256 + (logical % 256).long()
    w = flat_words[rows].view(length, 4, 8, 4).to(torch.int64) & 0xFFFFFFFF
    nibbles = (w[..., None] >> (4 * torch.arange(8, device=DEV))) & 15
    s = flat_scales[rows].view(length, 4, 8, 1, 1).float()
    return ((nibbles.float() - 7.5) * (s * 0.125)).view(length, 4, 256)


def reference_rotated(q, kc, ks, vc, vs, block_table, past, scale):
    total = past + q.shape[1]
    k = unpack_rotated(kc, ks, block_table[0], total).repeat_interleave(6, dim=1)
    v = unpack_rotated(vc, vs, block_table[0], total).repeat_interleave(6, dim=1)
    qr = (q.float().view(*q.shape[:3], 8, 32) @ h32_matrix()).view(q.shape)
    qr = qr * (scale * LOG2E * R32)
    outputs = []
    for qpos in range(q.shape[1]):
        scores = torch.einsum("bhd,nhd->bhn", qr[:, qpos], k)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = torch.exp2(scores)
        numerator = torch.einsum("bhn,nhd->bhd", probs, v)
        outputs.append(inverse_rotate(numerator / probs.sum(dim=-1, keepdim=True)))
    return torch.stack(outputs, dim=1)


def make_score_case(scores, splits=30, seed=77):
    scores = torch.as_tensor(scores, dtype=torch.float32, device=DEV)
    total = scores.numel()
    case = list(make_case(total, seed=seed, splits=splits))
    q, kc, ks = case[:3]
    q.zero_()
    q[:, :, 0::6, :].fill_(1.0)
    kc.zero_()
    ks.zero_()
    pages = case[5][0].long()
    rows = pages[ torch.arange(total, device=DEV) // 256 ] * 256 + torch.arange(total, device=DEV) % 256
    factor = 32 * (1 / 16) * LOG2E * R32 * (7.5 / 8)
    magnitudes = (scores.abs() / factor).half()
    positive = scores >= 0
    for h in range(4):
        kc.view(-1, 128)[rows, h * 32] = positive.to(torch.int32) * 15
        ks.view(-1, 32)[rows, h * 8] = magnitudes
    return case


@pytest.mark.parametrize("profile", [
    "constant", "decreasing", "increasing", "jump_below", "jump_exact", "jump_above",
    "underflow", "large_finite",
])
def test_softmax_profiles(profile):
    n = 1024
    if profile == "constant":
        scores = torch.zeros(n)
    elif profile == "decreasing":
        scores = 8 - 0.02 * torch.arange(n)
    elif profile == "increasing":
        scores = -8 + 0.02 * torch.arange(n)
    elif profile.startswith("jump"):
        jump = {"jump_below": 11.9, "jump_exact": 12.0, "jump_above": 12.1}[profile]
        scores = torch.zeros(n)
        scores[256:] = jump
    elif profile == "underflow":
        scores = torch.linspace(-160, -130, n)
        scores[768:] = 0
    else:
        scores = torch.full((n,), -100.0)
        scores[256:512] = 100.0
        scores[512:] = -80.0
    case = make_score_case(scores)
    out = run_native(case)
    q, kc, ks, vc, vs, bt, past = case[:7]
    ref = reference_rotated(q, kc, ks, vc, vs, bt, int(past.item()), 1 / 16)
    check_close(out, ref, profile)


def test_single_split_long_monotonic():
    scores = torch.linspace(-20, 20, 8326)
    case = make_score_case(scores, splits=1)
    out = run_native(case, splits=1)
    q, kc, ks, vc, vs, bt, past = case[:7]
    ref = reference_rotated(q, kc, ks, vc, vs, bt, int(past.item()), 1 / 16)
    check_close(out, ref, "single-split monotonic")


def test_decode_vs_prefill_and_eager_dispatch(monkeypatch):
    from exllamav3.modules.attention_fn.common import AttnArgs
    from exllamav3.modules.attention_fn.maxwell_q4 import fn_maxwell_q4_decode
    from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill

    case = make_case(258, seed=813, splits=30)
    q, kc, ks, vc, vs, bt, past = case[:7]
    common = dict(
        q=q, k=None, v=None, k_cache=kc, v_cache=vc, block_table=bt, cache_seqlens=past,
        causal=True, softmax_scale=1 / 16, qc=(ks, vs, 4, 4), pre_appended_len=1,
        n_kv_heads_override=4, max_kv_len=258,
    )
    prefill = paged_attn_triton_prefill(**common)
    args = AttnArgs(
        bsz=1, q_len=1, num_q_heads=24, dim=256, kv_len=1, num_kv_heads=4,
        q=q, k=q, v=q, k_cache=None, v_cache=None, causal=True, sm_scale=1 / 16,
        cu_seqlens=None, max_seqlen=None, window_size=None, softcap=0.0,
        block_table=bt, cache_seqlens=past, q_cache=(kc, ks, vc, vs, 4, 4),
    )
    monkeypatch.setenv("EXL3_ATTN_Q4M", "1")
    native = fn_maxwell_q4_decode(args)
    assert native is not None
    torch.testing.assert_close(native, prefill, atol=2e-3, rtol=0)
    assert torch.isfinite(native).all()
    monkeypatch.delenv("EXL3_ATTN_Q4M")
    assert fn_maxwell_q4_decode(args) is None


@pytest.mark.parametrize("total", [129, 257])
def test_sanitizer_native(total):
    case = make_case(total, seed=5000 + total, splits=30)
    out = run_native(case)
    assert torch.isfinite(out).all()
