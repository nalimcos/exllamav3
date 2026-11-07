"""cuBLAS chunked prefill attention for pre-Ampere quantized paged caches."""
import math
import os

import torch
import triton
import triton.language as tl

from .common import AttnArgs
from ...ext import exllamav3_ext as ext
from ...util.backend import ROCM

_pap_cublas = os.environ.get("EXL3_PAP_CUBLAS", "1") != "0"
_BLOCK_M = 256
_BLOCK_N = 256


def _cublas_prefill_available(device: torch.device) -> bool:
    return (
        _pap_cublas and not ROCM and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] < 8
    )


@triton.jit
def _softmax_rescale_kernel(
    scores, acc, running_m, running_l,
    n_cols, rows, q_len, q_start, n_start,
    FIRST: tl.constexpr,
    MASKED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_r = tl.program_id(0)
    batch = tl.program_id(1)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    n = tl.arange(0, BLOCK_N)
    d = tl.arange(0, HEAD_DIM)
    row_mask = r < rows
    col_mask = n < n_cols
    mask = row_mask[:, None] & col_mask[None, :]
    store_mask = mask

    s_offsets = (batch * rows + r[:, None]) * n_cols + n[None, :]
    s = tl.load(scores + s_offsets, mask=mask, other=-float("inf"))
    if MASKED:
        q_pos = q_start + r % q_len
        valid = n[None, :] + n_start <= q_pos[:, None]
        s = tl.where(valid, s, -float("inf"))
        mask &= valid

    tile_m = tl.max(s, axis=1)
    if FIRST:
        m_new = tile_m
        alpha = tl.full((BLOCK_R,), 1.0, tl.float32)
        old_l = tl.zeros((BLOCK_R,), tl.float32)
    else:
        old_m = tl.load(running_m + batch * rows + r, mask=row_mask, other=0.0)
        old_l = tl.load(running_l + batch * rows + r, mask=row_mask, other=0.0)
        m_new = tl.maximum(old_m, tile_m)
        alpha = tl.exp(old_m - m_new)

    p = tl.exp(s - m_new[:, None])
    p = tl.where(mask, p, 0.0)
    tl.store(scores + s_offsets, p, mask=store_mask)
    tl.store(running_m + batch * rows + r, m_new, mask=row_mask)
    tl.store(running_l + batch * rows + r, old_l * alpha + tl.sum(p, axis=1), mask=row_mask)

    if not FIRST:
        a_offsets = (batch * rows + r[:, None]) * HEAD_DIM + d[None, :]
        a = tl.load(acc + a_offsets, mask=row_mask[:, None], other=0.0)
        tl.store(acc + a_offsets, a * alpha[:, None], mask=row_mask[:, None])


def paged_attn_cublas_prefill(
    q: torch.Tensor,
    qk: torch.Tensor,
    qv: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    qc: tuple,
    pre_appended_len: int,
    n_kv_heads: int,
    past: int,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Batch-1 causal Q4 attention over pre-appended cache rows.

    Q/K and P/V are FP32 batched SGEMMs. Softmax and output rescaling stay FP32.
    The dispatcher establishes the remaining shape constraints.
    """
    bsz, q_len, n_q_heads, head_dim = q.shape
    assert bsz == 1 and q_len == pre_appended_len
    assert head_dim == 256 and q.dtype == torch.float16 and q.is_contiguous()
    assert n_q_heads % n_kv_heads == 0
    page_size = qk.shape[1]
    total = past + q_len
    pages = triton.cdiv(total, page_size)
    sk, sv, k_bits, v_bits = qc
    assert k_bits == 4 and v_bits == 4
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Dequantize once per layer/chunk. The extension writes [0, total) densely and
    # unrotates K/V. Drop each fp16 buffer after conversion to bound peak scratch.
    kd = torch.empty((pages, page_size, n_kv_heads, head_dim), dtype=torch.half, device=q.device)
    vd = torch.empty_like(kd)
    ext.dequant_cache_paged_window(
        qk, sk, kd, qv, sv, vd, cache_seqlens, block_table[:, :pages].contiguous(),
        page_size, q_len, 0.0,
    )
    k = kd.view(-1, n_kv_heads, head_dim)[:total].permute(1, 0, 2).to(
        dtype=torch.float32, memory_format=torch.contiguous_format)
    del kd
    v = vd.view(-1, n_kv_heads, head_dim)[:total].permute(1, 0, 2).to(
        dtype=torch.float32, memory_format=torch.contiguous_format)
    del vd

    group = n_q_heads // n_kv_heads
    max_rows = n_q_heads * _BLOCK_M
    score_storage = torch.empty(max_rows * _BLOCK_N, dtype=torch.float32, device=q.device)
    acc_storage = torch.empty(max_rows * head_dim, dtype=torch.float32, device=q.device)
    m_storage = torch.empty(max_rows, dtype=torch.float32, device=q.device)
    l_storage = torch.empty(max_rows, dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)

    for q0 in range(0, q_len, _BLOCK_M):
        q1 = min(q0 + _BLOCK_M, q_len)
        m_len = q1 - q0
        rows_per_batch = group * m_len
        rows = n_kv_heads * rows_per_batch
        # Pack the GQA siblings into rows of four batched (kv_head) matrices.
        q_chunk = q[0, q0:q1].permute(1, 0, 2).to(
            dtype=torch.float32, memory_format=torch.contiguous_format)
        q_chunk.mul_(softmax_scale)
        q_chunk = q_chunk.view(n_kv_heads, rows_per_batch, head_dim)
        acc = acc_storage[:rows * head_dim].view(n_kv_heads, rows_per_batch, head_dim)
        running_m = m_storage[:rows].view(n_kv_heads, rows_per_batch)
        running_l = l_storage[:rows].view(n_kv_heads, rows_per_batch)
        q_start = past + q0
        kv_end = past + q1
        first = True

        for n0 in range(0, kv_end, _BLOCK_N):
            n1 = min(n0 + _BLOCK_N, kv_end)
            n_len = n1 - n0
            scores = score_storage[:rows * n_len].view(n_kv_heads, rows_per_batch, n_len)
            torch.bmm(q_chunk, k[:, n0:n1].transpose(1, 2), out=scores)
            _softmax_rescale_kernel[(triton.cdiv(rows_per_batch, 4), n_kv_heads)](
                scores, acc, running_m, running_l,
                n_len, rows_per_batch, m_len, q_start, n0,
                FIRST=first, MASKED=n1 > q_start, HEAD_DIM=head_dim,
                BLOCK_N=triton.next_power_of_2(n_len), BLOCK_R=4,
                num_warps=4, num_stages=1,
            )
            if first:
                torch.bmm(scores, v[:, n0:n1], out=acc)
                first = False
            else:
                acc.baddbmm_(scores, v[:, n0:n1])

        acc.div_(running_l.unsqueeze(-1))
        out[0, q0:q1].copy_(acc.view(n_q_heads, m_len, head_dim).permute(1, 0, 2))

    return out


def fn_cublas_paged_attn_prefill_qc(args: AttnArgs) -> torch.Tensor | None:
    if (
        not _cublas_prefill_available(args.q.device)
        or args.q_cache is None
        or args.is_varlen()
        or args.bsz != 1
        or args.q_len <= 16
        or args.dim != 256
        or args.q.dtype != torch.float16
        or not args.q.is_contiguous()
        or not args.causal
        or args.get_window_size() != (-1, -1)
        or args.softcap
        or args.sinks is not None
        or args.sink_key0
        or args.non_causal_spans
        or args.max_kv_len is None
        or args.num_q_heads % args.num_kv_heads
    ):
        return None
    qk, sk, qv, sv, k_bits, v_bits = args.q_cache
    if k_bits != 4 or v_bits != 4:
        return None
    return paged_attn_cublas_prefill(
        q=args.q, qk=qk, qv=qv,
        block_table=args.block_table,
        cache_seqlens=args.cache_seqlens,
        qc=(sk, sv, k_bits, v_bits),
        pre_appended_len=args.q_len,
        n_kv_heads=args.num_kv_heads,
        past=args.max_kv_len,
        softmax_scale=args.sm_scale,
    )
