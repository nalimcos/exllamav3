import os
import torch

# Turing (sm_75) fast paths. GeForce Turing parts have no bf16 tensor cores, run fp32-accumulate HMMA at
# half the fp16-accumulate rate, give a block 64 KB of shared memory, and Triton lowers tl.dot to scalar
# FMA there, so generic paths can leave most of the chip idle. Each path below is switched by an
# EXL3_<NAME> environment variable; when the variable is unset, it defaults to the value here on sm_75
# devices and to 0 (upstream behaviour) on every other architecture.

SM75_DEFAULTS = {
    "SDPA_PREFILL": 1,    # prefill attention on the dequantized window (PyTorch SDPA / fa75), not the packed cache
    "FA75": 1,            # flash-attention prefill kernel for head_dim 256 (needs SDPA_PREFILL)
}

_cc_cache = {}

def _device_index(device) -> int:
    if device is None:
        return torch.cuda.current_device()
    elif isinstance(device, torch.Tensor):
        return device.device.index
    elif isinstance(device, torch.device):
        return device.index if device.index is not None else torch.cuda.current_device()
    elif isinstance(device, str):
        d = torch.device(device)
        return d.index if d.index is not None else torch.cuda.current_device()
    else:
        return int(device)


def _capability(device) -> tuple[int, int]:
    idx = _device_index(device)
    cc = _cc_cache.get(idx)
    if cc is None:
        cc = torch.cuda.get_device_capability(idx) if torch.version.cuda else (0, 0)
        _cc_cache[idx] = cc
    return cc


def _gemv_core(device) -> int:
    # Per-device probe owned by the extension: 1 means the CUDA-core inner loop was selected, i.e.
    # the part has no usable tensor cores (sm_75 GeForce / TU117). Reused here rather than adding a
    # second heuristic. The extension caches the decision; it is resolved on the first uncaptured
    # call, so callers may transiently see 0 while a graph capture is in progress.
    try:
        idx = _device_index(device)
        from ..ext import exllamav3_ext as ext
        return 1 if ext.g_get_gemv_core(idx) == 1 else 0
    except Exception:
        return 0


def turing_flag(name: str, device = None) -> int:
    """
    Level of the sm_75 fast path `name` for `device`: EXL3_<name> if set (0 disables, any other integer
    is taken as given), else SM75_DEFAULTS[name] on sm_75 and 0 elsewhere.
    """
    env = os.environ.get("EXL3_" + name)
    if env is not None:
        try:
            return int(env)
        except ValueError:
            return 0 if env.strip().lower() in ("", "false", "off", "no") else 1
    if not torch.cuda.is_available():
        return 0
    if _capability(device) != (7, 5):
        return 0
    default = SM75_DEFAULTS.get(name, 0)
    if name == "FA75" and default:
        # fa75 emits mma.sync.m16n8k8; sm_75 parts without usable tensor cores (TU117 / GeForce
        # GTX 16-series) run those on the CUDA cores, where fa75 is slower than the SDPA path.
        if _gemv_core(device) == 1:
            return 0
    return default
