"""
No-GPU sm_50/52 JIT compile gate for the Maxwell attention kernels (Workstream D).

Compiles the worktree's real production Triton attention kernels (paged decode split /
combine / prefill) for Maxwell CC 5.0 and 5.2 via an explicit target override -- no GPU,
no arch-list -- and asserts each produces a real cubin with ZERO tensor-core (`mma`) ops.
On Maxwell, `tl.dot` must lower to fp32 FMA; any `mma` in the PTX means a tensor-core
instruction was emitted, which Maxwell (CC 5.0/5.2) cannot execute.

This re-verifies the Workstream-A make-or-break verdict for the attention path and doubles
as a permanent gate. Pure host: it only JIT-compiles, it never launches a kernel.

    python tests/triton_maxwell_compile_check.py
Exit 0 on all-PASS, nonzero on any FAIL.
"""

import os
import re
import sys

# Standalone: put the worktree root (parent of tests/) on the path so `exllamav3` resolves.
_WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKTREE not in sys.path:
    sys.path.insert(0, _WORKTREE)

from triton import compile as tcompile
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

import exllamav3.modules.attention_fn.triton_paged as TP

# Each entry: kernel, triton signature (tensor dtypes), constexprs, and launch options.
# Signatures/constexprs match the production call sites (head_dim 256, GQA 8, 4-bit KV).
KERNELS = [
    {
        "name": "decode_split",
        "kernel": TP._paged_attn_decode_split_kernel,
        "sig": {
            "q": "*fp16", "k_cache": "*i32", "v_cache": "*i32", "block_table": "*i32",
            "cache_seqlens": "*i32", "out": "*fp16", "partial_o": "*fp32", "partial_ml": "*fp32",
            "k_scales": "*fp16", "v_scales": "*fp16", "h32": "*fp16",
            "split_len": "i32", "num_pages_per_seq": "i32", "num_splits": "i32", "sinks": "*fp32",
        },
        "const": {
            "QCK": 4, "QCV": 4, "q_len": 1, "kv_append_len": 1, "n_q_heads": 64, "n_kv_heads": 8,
            "page_size": 256, "head_dim": 256, "HD_PAD": 256, "scale": 0.125, "CAUSAL": True,
            "WINDOW_LEFT": -1, "WINDOW_RIGHT": 0, "SOFTCAP": 0.0, "FINAL": False,
            "HAS_SINKS": False, "BLOCK_ROWS": 8, "BLOCK_N": 32,
        },
        "num_warps": 4, "num_stages": 1,
    },
    {
        "name": "decode_combine",
        "kernel": TP._paged_attn_decode_combine_kernel,
        "sig": {
            "partial_o": "*fp32", "partial_ml": "*fp32", "out": "*fp16", "h32": "*fp16",
            "num_splits": "i32", "sinks": "*fp32",
        },
        "const": {
            "QCV": 4, "HAS_SINKS": False, "q_len": 1, "n_q_heads": 64, "n_kv_heads": 8,
            "head_dim": 256, "HD_PAD": 256, "V_DIM": 256, "BLOCK_ROWS": 8, "ROWS_SUB": 1, "D_SUB": 256,
        },
        "num_warps": 4, "num_stages": 1,
    },
    {
        "name": "prefill",
        "kernel": TP._paged_attn_prefill_kernel,
        "sig": {
            "q": "*fp16", "k_cache": "*i32", "v_cache": "*i32", "block_table": "*i32",
            "cache_seqlens": "*i32", "out": "*fp16", "partial_o": "*fp32", "partial_ml": "*fp32",
            "k_scales": "*fp16", "v_scales": "*fp16", "h32": "*fp16", "k_new": "*fp16",
            "v_new": "*fp16", "sinks": "*fp32", "num_splits": "i32",
            "q_len": "i32", "kv_append_len": "i32", "num_pages_per_seq": "i32",
            "WINDOW_LEFT": "i32", "WINDOW_RIGHT": "i32",
        },
        "const": {
            "IS_SPLIT": False, "NEW_KV": 1, "QCK": 4, "QCV": 4, "n_q_heads": 64, "n_kv_heads": 8,
            "page_size": 256, "head_dim": 256, "HD_PAD": 256, "scale": 0.125, "CAUSAL": True,
            "HAS_WINDOW_LEFT": False, "HAS_WINDOW_RIGHT": False, "SOFTCAP": 0.0,
            "HAS_SINKS": False, "WIDE_INDEX": False, "BLOCK_M": 64, "BLOCK_N": 64,
        },
        "num_warps": 8, "num_stages": 2,
    },
]

ARCHS = [("sm_50", 50), ("sm_52", 52)]


def check_one(kernel_spec, arch, cc):
    """Compile one kernel for one arch; return (ok, message)."""
    src = ASTSource(fn=kernel_spec["kernel"], signature=kernel_spec["sig"], constexprs=kernel_spec["const"])
    target = GPUTarget("cuda", cc, 32)
    options = {
        "num_warps": kernel_spec["num_warps"],
        "num_stages": kernel_spec["num_stages"],
        "num_ctas": 1,
    }
    try:
        c = tcompile(src, target=target, options=options)
    except Exception as e:
        return False, f"compile error: {e!r}"

    cubin = c.asm.get("cubin", b"")
    ptx = c.asm.get("ptx", "")
    targets = set(re.findall(r"\.target\s+\S+", ptx))
    mma = len(re.findall(r"\bmma\.", ptx))

    problems = []
    if len(cubin) == 0:
        problems.append("empty cubin")
    if f".target sm_{cc}" not in targets:
        problems.append(f"PTX target {targets} != sm_{cc}")
    if mma != 0:
        problems.append(f"{mma} mma (tensor-core) ops")

    ok = not problems
    detail = f"cubin={len(cubin)}B target={sorted(targets)} mma={mma}"
    return ok, (detail if ok else f"{detail} -> {', '.join(problems)}")


def main():
    results = []
    for spec in KERNELS:
        for arch, cc in ARCHS:
            ok, msg = check_one(spec, arch, cc)
            tag = "PASS" if ok else "FAIL"
            line = f"{tag}  {spec['name']:<14} {arch}  {msg}"
            print(line)
            results.append(ok)

    all_ok = all(results)
    print()
    print(f"{'ALL PASS' if all_ok else 'FAILURES PRESENT'}  ({sum(results)}/{len(results)} kernels compiled clean)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
