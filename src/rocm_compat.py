"""ROCm / AMD GPU compatibility helpers for GSQ.

GSQ's quantization math is plain PyTorch, so most of it runs unchanged on
ROCm.  This module isolates the places where the original implementation
assumed NVIDIA hardware:

* the CUDA RNG capture used by the custom Gumbel-Softmax autograd op
  (``torch.cuda.get/set_rng_state``) -- also makes the op work on CPU;
* device-name normalisation (``cuda`` is still the PyTorch device namespace
  on ROCm, but users often write ``rocm``/``hip``);
* architecture inspection (``gcnArchName`` instead of ``sm_XX``);
* CUDA-only flags (TF32, cuDNN) and CUDA-only kernels (Humming / Marlin);
* flash-attention fallbacks.

Everything here is defensive: on a CUDA build it returns exactly the same
behaviour as the original code.
"""

from __future__ import annotations

import contextlib
import os
import warnings
from typing import Optional, Union

import torch

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

IS_ROCM = getattr(torch.version, "hip", None) is not None
HIP_VERSION = getattr(torch.version, "hip", None)
CUDA_VERSION = getattr(torch.version, "cuda", None)
IS_CUDA = (not IS_ROCM) and CUDA_VERSION is not None
GPU_AVAILABLE = torch.cuda.is_available()

DeviceLike = Union[str, torch.device, int, None]


def is_rocm() -> bool:
    return IS_ROCM


def is_gpu_available() -> bool:
    return torch.cuda.is_available()


def normalize_device(device: DeviceLike = None) -> torch.device:
    """Return a valid torch device.

    ROCm builds expose GPUs through the historical ``cuda`` namespace, so
    ``rocm`` / ``hip`` / ``amd`` / ``gpu`` are accepted as aliases.  Missing
    GPUs fall back to CPU instead of raising, which keeps CPU smoke tests
    working on machines without a GPU.
    """
    if device is None:
        return torch.device("cuda") if is_gpu_available() else torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device("cuda", device) if is_gpu_available() else torch.device("cpu")

    spec = str(device).strip()
    lowered = spec.lower()
    aliases = ("rocm", "hip", "amd", "gpu")
    for alias in aliases:
        if lowered == alias:
            spec = "cuda"
            break
        if lowered.startswith(alias + ":"):
            spec = "cuda:" + spec.split(":", 1)[1]
            break

    dev = torch.device(spec)
    if dev.type == "cuda" and not is_gpu_available():
        return torch.device("cpu")
    return dev


def default_device() -> str:
    """Device string used by the entry points (``cuda`` under ROCm)."""
    return "cuda" if is_gpu_available() else "cpu"


def device_index(device: DeviceLike = None) -> Optional[int]:
    dev = normalize_device(device)
    if dev.type != "cuda" or not is_gpu_available():
        return None
    if dev.index is not None:
        return int(dev.index)
    return int(torch.cuda.current_device())


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------


def _props(index: int = 0):
    if not is_gpu_available():
        return None
    try:
        return torch.cuda.get_device_properties(index)
    except Exception:
        return None


def arch_name(index: Optional[int] = None) -> str:
    """Return ``gfxXXXX`` on ROCm, or ``sm_XX``-style info on CUDA.

    ``torch.cuda.get_device_properties().gcnArchName`` only exists on ROCm.
    """
    if index is None:
        index = device_index() or 0
    props = _props(index)
    if props is None:
        return "cpu"
    gcn = getattr(props, "gcnArchName", None)
    if gcn:
        # ROCm may append feature flags: "gfx1200:sramecc+:xnack-"
        return str(gcn).split(":", 1)[0]
    if IS_ROCM:
        return "unknown-gfx"
    major, minor = torch.cuda.get_device_capability(index)
    return f"sm_{major}{minor}"


#: Discrete RDNA4 (RX 9000 class).
RDNA4_ARCHES = ("gfx1200", "gfx1201")
#: RDNA3.5 APUs: Strix Point / Strix Halo / Krackan Point / gfx1153.
RDNA35_ARCHES = ("gfx1150", "gfx1151", "gfx1152", "gfx1153")
SUPPORTED_ARCHES = RDNA4_ARCHES + RDNA35_ARCHES


def is_gfx12(index: Optional[int] = None) -> bool:
    """True for RDNA4 (gfx1200/gfx1201)."""
    return arch_name(index).startswith(RDNA4_ARCHES)


def is_rdna4(index: Optional[int] = None) -> bool:
    return is_gfx12(index)


def is_rdna35(index: Optional[int] = None) -> bool:
    """True for RDNA3.5 APUs (gfx1150/gfx1151/gfx1152/gfx1153)."""
    return arch_name(index).startswith(RDNA35_ARCHES)


def is_apu(index: Optional[int] = None) -> bool:
    """RDNA3.5 targets are Ryzen APUs with unified (GTT) memory."""
    return is_rdna35(index)


def gfx_family(arch: Optional[str] = None) -> str:
    arch = arch or arch_name()
    if arch.startswith(RDNA4_ARCHES):
        return "RDNA4"
    if arch.startswith(RDNA35_ARCHES):
        return "RDNA3.5-APU"
    if arch.startswith(("gfx11",)):
        return "RDNA3"
    if arch.startswith(("gfx10",)):
        return "RDNA2/1"
    return "unknown"


def describe_devices() -> str:
    if not is_gpu_available():
        return "no GPU (CPU mode)"
    lines = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024 ** 3
        lines.append(f"[{i}] {props.name} ({arch_name(i)}, {mem_gb:.1f} GiB)")
    return "; ".join(lines)


def warn_environment() -> None:
    """Print a one-time platform banner and ROCm-specific advisories."""
    if not is_gpu_available():
        warnings.warn(
            "No GPU detected. GSQ will run on CPU, which is only intended for "
            "small smoke tests.",
            RuntimeWarning,
        )
        return
    if IS_ROCM:
        arches = {arch_name(i) for i in range(torch.cuda.device_count())}
        unsupported = sorted(a for a in arches if not a.startswith(SUPPORTED_ARCHES))
        if unsupported:
            warnings.warn(
                f"ROCm build active but GPUs are {sorted(arches)}. This port was "
                "validated for gfx1200/gfx1201 (RDNA4) and gfx115x (RDNA3.5 APUs); "
                "other architectures should still work but are untested.",
                RuntimeWarning,
            )
        if any(a.startswith(RDNA35_ARCHES) for a in arches):
            warnings.warn(
                "RDNA3.5 APU detected (gfx1150/gfx1151/gfx1152/gfx1153). The GPU "
                "has no dedicated VRAM: it maps system RAM through GTT. Keep the "
                "BIOS VRAM carve-out small and raise the TTM/GTT limit if a model "
                "does not fit (e.g. 'amd-ttm --set 32'). gfx1152 (Krackan Point) "
                "needs ROCm >= 7.2. GSQ's per-layer streaming is the safest mode "
                "on these parts.",
                RuntimeWarning,
            )



def prefer_sdpa(requested: Optional[str] = None, strict: bool = False) -> str:
    """Resolve an attention implementation that exists on this stack.

    ``flash_attention_2`` is a CUDA-only extension and cannot be built for
    ROCm/gfx1200.  Mirrors that use it crash at model construction, so map it
    to ``sdpa`` (PyTorch native, works on ROCm) unless ``strict`` is set.
    """
    impl = (requested or "").strip().lower()
    if impl in ("", "auto", "none"):
        return "sdpa"
    if impl == "flash_attention_2" and IS_ROCM:
        msg = (
            "flash_attention_2 is not available on ROCm. Use --attn-implementation "
            "sdpa (recommended) or eager."
        )
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning)
        return "sdpa"
    return impl


# ---------------------------------------------------------------------------
# CUDA-only feature gates
# ---------------------------------------------------------------------------


def humming_supported() -> bool:
    """Humming / MARLIN kernels are CUDA-only (nvcc + PTX intrinsics)."""
    return not IS_ROCM


def require_humming(feature: str = "Humming kernels") -> None:
    # Explicit escape hatch for someone who has a HIP port of the kernels.
    if os.environ.get("GSQ_ALLOW_HUMMING", "").lower() in ("1", "true", "yes"):
        return
    if not humming_supported():
        raise RuntimeError(
            f"{feature} are CUDA-only and cannot run on ROCm/gfx1200. "
            "Use the compressed-tensors / transformers dequantize path instead. "
            "Set GSQ_ALLOW_HUMMING=1 only if you have a HIP port of the kernels."
        )


# ---------------------------------------------------------------------------
# RNG helpers (custom autograd Function needs exact forward RNG replay)
# ---------------------------------------------------------------------------


def get_rng_state(device: DeviceLike = None):
    idx = device_index(device)
    if idx is None:
        return torch.get_rng_state()
    return torch.cuda.get_rng_state(device=idx)


def set_rng_state(state, device: DeviceLike = None) -> None:
    idx = device_index(device)
    if idx is None:
        torch.set_rng_state(state)
        return
    torch.cuda.set_rng_state(state, device=idx)


@contextlib.contextmanager
def fork_rng(device: DeviceLike = None):
    """Fork RNG state around a block, including CPU and ROCm devices."""
    idx = device_index(device)
    if idx is None:
        with torch.random.fork_rng(devices=[], enabled=True):
            yield
    else:
        with torch.random.fork_rng(devices=[idx], enabled=True):
            yield


def rand_like(tensor: torch.Tensor) -> torch.Tensor:
    """``torch.rand_like`` with a safe half-precision path.

    Some ROCm builds may lack a half-precision random kernel for a specific
    gfx target.  ``GSQ_RNG_FP32=1`` samples in fp32 and casts as a fallback.
    Native sampling (the upstream behaviour) is the default.
    """
    force_fp32 = os.environ.get("GSQ_RNG_FP32")
    use_fp32 = (
        tensor.dtype in (torch.bfloat16, torch.float16)
        and force_fp32 in ("1", "true", "yes")
    )
    if use_fp32:
        return torch.rand(tensor.shape, dtype=torch.float32, device=tensor.device).to(tensor.dtype)
    return torch.rand_like(tensor)


# ---------------------------------------------------------------------------
# Small runtime wrappers
# ---------------------------------------------------------------------------


def empty_cache() -> None:
    if is_gpu_available():
        torch.cuda.empty_cache()


def synchronize(device: DeviceLike = None) -> None:
    if is_gpu_available():
        torch.cuda.synchronize(device)


def max_memory_allocated_gb() -> float:
    if not is_gpu_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def max_memory_reserved_gb() -> float:
    if not is_gpu_available():
        return 0.0
    return torch.cuda.max_memory_reserved() / (1024 ** 3)


def reset_peak_memory_stats() -> None:
    if is_gpu_available():
        torch.cuda.reset_peak_memory_stats()


def dist_backend() -> str:
    """RCCL is exposed as ``nccl`` by PyTorch ROCm; allow an override."""
    return os.environ.get("GSQ_DIST_BACKEND", "nccl")


def disable_tf32() -> None:
    """Disable TF32 where the backend has such a flag.

    ROCm has no TF32; the flags are either absent or no-ops.  Guarding them
    avoids attribute errors on older ROCm builds.
    """
    for path in ("cuda.matmul.allow_tf32", "cudnn.allow_tf32"):
        try:
            obj = torch.backends
            parts = path.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], False)
        except Exception:  # attribute missing or guarded on this backend
            pass


def environment_summary() -> str:
    if IS_ROCM:
        backend = f"ROCm {HIP_VERSION}"
    elif IS_CUDA:
        backend = f"CUDA {CUDA_VERSION}"
    else:
        backend = "CPU"
    return (
        f"torch={torch.__version__} backend={backend} "
        f"devices={describe_devices()}"
    )
