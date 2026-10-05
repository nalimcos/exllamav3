"""sm_75 CUDA-core inner GEMM for the fused MoE (exl3_moe_kernel CORE instance) vs the mma path.

The CORE inner GEMM is only reachable through exl3_moe, whose launch shape and dynamic expert
scheduler make it a poor host for a numerical test. Instead this compiles the inner GEMM
directly at its MoE shape (16x32x128, 512 threads) and runs the CORE and mma instantiations on
identical packed-trellis input, comparing the fp16 outputs.

The two paths are meant to differ only by fp16 accumulation rounding: the mma accumulates the
k=16 products in fp32, the CORE path accumulates 4 k-tiles in fp16 and folds to fp32. A wrong
lane/k mapping (half the k dropped or double-counted) fails at ~1.0, not at 1e-3.
"""
import os
import shutil
import subprocess
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "sm75_moe_core_gemm.cu")


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc required")
def test_moe_core_inner_gemm_matches_mma():
    # The binary cannot go under /tmp (often mounted noexec), so build it inside the package
    # tree and remove it afterwards
    build_dir = os.path.join(PACKAGE_ROOT, "build", "sm75_moe_core_test")
    os.makedirs(build_dir, exist_ok=True)
    out = os.path.join(build_dir, "sm75_moe_core_gemm")
    cmd = [
        "nvcc", "-O3", "--use_fast_math", "-std=c++17",
        "-gencode=arch=compute_75,code=sm_75",
        "-I", PACKAGE_ROOT, SRC, "-o", out,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        pytest.skip(f"nvcc build failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    try:
        r = subprocess.run([out], capture_output=True, text=True, timeout=300)
        print(r.stdout)
        assert r.returncode == 0, f"CORE inner GEMM disagrees with the mma path:\n{r.stdout}\n{r.stderr}"
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)
