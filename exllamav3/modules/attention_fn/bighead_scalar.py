import torch
from ...ext import exllamav3_ext as ext
from .common import AttnArgs, AttnFn, get_non_causal_span_arglist

KV_CHUNK_SIZE = 128

def fn_bighead_scalar_attn(args: AttnArgs) -> torch.Tensor | None:
    # Native-CUDA fp32 fallback for the dims the kernel instantiates
    # ({64,128,256} x G∈{1,2,4,8}, {512} x G∈{1,2,4,8,16}), used after Triton
    # for both decode and prefill. Declines non-instantiated dims / GQA ratios
    # here so they never reach the CUDA TORCH_CHECK.
    g = args.num_q_heads // args.num_kv_heads if args.num_kv_heads else None
    allowed_g = (1, 2, 4, 8, 16) if args.dim == 512 else (1, 2, 4, 8)
    if (
        args.is_varlen() or
        args.is_swa() or
        not args.has_kv_cache() or
        args.dim not in (64, 128, 256, 512) or
        g not in allowed_g or
        args.softcap != 0.0 or
        args.sinks is not None or
        args.non_causal_spans
    ):
        return None

    if not args.non_causal_spans:
        return _scalar_bighead_fallback(
            q = args.q,
            k = args.k,
            v = args.v,
            k_cache = args.k_cache,
            v_cache = args.v_cache,
            block_table = args.block_table,
            cache_seqlens = args.cache_seqlens,
            causal = args.causal,
            softmax_scale = args.sm_scale,
            window_size = args.get_window_size(),
            softcap = args.softcap
        )
    else:
        arglist = get_non_causal_span_arglist(args)
        o = [_scalar_bighead_fallback(**a) for a in arglist]
        return torch.cat(o, dim = 1)


def _scalar_bighead_fallback(
    q, k, v,
    k_cache, v_cache,
    block_table, cache_seqlens,
    causal = False, softmax_scale = None,
    window_size = None,
    softcap = None,
    kv_chunk_size = 128,
):
    o = torch.empty_like(q)
    ext.bighead_attn_paged(
        q, k, v,
        k_cache, v_cache,
        block_table, cache_seqlens,
        o,
        kv_chunk_size,
        causal,
        softmax_scale
    )
    return o
