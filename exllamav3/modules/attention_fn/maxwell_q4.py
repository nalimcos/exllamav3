"""Native Maxwell Q4 decode attention for the fixed 24Q/4KV, D=256 shape."""
import functools
import os
import torch

from ...ext import exllamav3_ext as ext
from .common import AttnArgs


def _enabled() -> bool:
    return os.environ.get("EXL3_ATTN_Q4M", "").lower() in ("1", "true")


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0


@functools.lru_cache(maxsize=None)
def _resources(device: int):
    with torch.cuda.device(device):
        return tuple(ext.q4m_resource_info())


def fn_maxwell_q4_decode(args: AttnArgs) -> torch.Tensor | None:
    if not _enabled() or args.q_cache is None:
        return None
    if (
        args.bsz != 1 or args.q_len != 1 or
        args.num_q_heads != 24 or args.num_kv_heads != 4 or args.dim != 256 or
        args.q.dtype != torch.float16 or not args.q.is_contiguous() or
        args.causal is not True or args.is_varlen() or args.is_swa() or
        args.window_right != 0 or args.softcap != 0.0 or
        args.sinks is not None or args.sink_key0 or args.non_causal_spans is not None or
        torch.cuda.get_device_capability(args.q.device) != (5, 2)
    ):
        return None

    qk, sk, qv, sv, k_bits, v_bits = args.q_cache
    tensors = (qk, sk, qv, sv, args.block_table, args.cache_seqlens)
    if (
        k_bits != 4 or v_bits != 4 or
        qk.dtype != torch.int32 or qv.dtype != torch.int32 or
        sk.dtype != torch.float16 or sv.dtype != torch.float16 or
        args.block_table.dtype != torch.int32 or args.cache_seqlens.dtype != torch.int32 or
        any(t is None or not t.is_contiguous() or t.device != args.q.device for t in tensors) or
        qk.ndim != 3 or qk.shape[1:] != (256, 128) or qv.shape != qk.shape or
        sk.shape != (qk.shape[0], 256, 32) or sv.shape != sk.shape or
        args.block_table.ndim != 2 or args.block_table.shape[0] != 1 or
        args.cache_seqlens.numel() != 1 or
        not all(_aligned(t) for t in (args.q, qk, sk, qv, sv, args.block_table))
    ):
        return None

    resources = _resources(args.q.device.index)
    if resources[2] > 96 or resources[3] > 16672 or resources[4] != 0 or resources[5] < 5:
        return None

    splits = 30
    out = torch.empty_like(args.q)
    q_rot = torch.empty(args.q.shape, dtype=torch.float32, device=args.q.device)
    partial_o = torch.empty(4 * splits * 6 * 256, dtype=torch.float32, device=args.q.device)
    partial_ml = torch.empty(4 * splits * 6 * 2, dtype=torch.float32, device=args.q.device)
    ext.q4m_attention(
        args.q, qk, sk, qv, sv, args.block_table, args.cache_seqlens,
        out, q_rot, partial_o, partial_ml, args.sm_scale, splits,
    )
    return out
