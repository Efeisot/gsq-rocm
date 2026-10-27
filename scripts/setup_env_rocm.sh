#!/usr/bin/env bash
# ============================================================================
# GSQ — ROCm / AMD (gfx1200 / gfx1201 / gfx1152) environment setup
# ============================================================================
# The CUDA path (`scripts/setup_env.sh`) installs torch+cu130 from uv.lock and
# flash-attn. Neither works on ROCm:
#   * the CUDA torch wheel cannot see an AMD GPU;
#   * flash-attn has no ROCm backend for RDNA4/RDNA3.5;
#   * vllm==0.20.2 in pyproject.toml is a CUDA wheel.
# This script installs a ROCm torch + the portable Python deps instead.
#
# Usage:
#   bash scripts/setup_env_rocm.sh
#   ROCM_WHEEL=rocm7.2 bash scripts/setup_env_rocm.sh   # override wheel tag
#   PYTHON_VERSION=3.12 VENV_PATH=/data/venvs/gsq-rocm bash scripts/setup_env_rocm.sh
#   INSTALL_VLLM=1 bash scripts/setup_env_rocm.sh        # optionally add vLLM ROCm
#
# Requirements:
#   * gfx1200/gfx1201: ROCm >= 6.4.2; ROCm 7.0+ recommended.
#   * gfx1150/gfx1151/gfx1152/gfx1153 (RDNA3.5 APUs): ROCm >= 7.2 on Linux
#     (kernel fixes; see ROCm's RDNA3.5 system-optimization page).
#   * A Python 3.11/3.12 interpreter.
#   * uv: https://docs.astral.sh/uv/getting-started/installation/
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv-rocm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_VLLM="${INSTALL_VLLM:-0}"
TORCH_VERSION="${TORCH_VERSION:-}"

# Pick a PyTorch ROCm wheel tag. gfx1152 (Krackan Point) is RDNA3.5 and needs
# ROCm >= 7.2; gfx120x works with 7.0+. A user-supplied ROCM_WHEEL wins.
DETECTED_GFX="$(rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*' || true)"
if [[ -z "${ROCM_WHEEL:-}" ]]; then
    case "${DETECTED_GFX}" in
        gfx115*) ROCM_WHEEL="rocm7.2" ;;
        *)       ROCM_WHEEL="rocm7.0" ;;
    esac
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed. Install with:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if ! command -v rocminfo >/dev/null 2>&1 && [[ ! -d /opt/rocm ]]; then
    echo "WARNING: no ROCm installation found (/opt/rocm missing, rocminfo not on PATH)." >&2
    echo "         Continuing anyway; the wheel install may still succeed." >&2
fi

echo "=========================================="
echo "GSQ — ROCm environment setup"
echo "Repo root  : ${REPO_ROOT}"
echo "Venv path  : ${VENV_PATH}"
echo "Python     : ${PYTHON_VERSION}"
echo "Wheel tag  : ${ROCM_WHEEL} (https://download.pytorch.org/whl/${ROCM_WHEEL})"
echo "vLLM       : $([ "${INSTALL_VLLM}" = "1" ] && echo INSTALL || echo SKIP)"
echo "=========================================="

cd "${REPO_ROOT}"
export UV_PROJECT_ENVIRONMENT="${VENV_PATH}"

echo "[1/4] Creating venv..."
uv venv --python "${PYTHON_VERSION}" "${VENV_PATH}"

echo "[2/4] Installing torch/torchvision from the ROCm wheel index..."
TORCH_SPEC="torch"
[[ -n "${TORCH_VERSION}" ]] && TORCH_SPEC="torch==${TORCH_VERSION}"
uv pip install --python "${VENV_PATH}/bin/python" \
    --index-url "https://download.pytorch.org/whl/${ROCM_WHEEL}" \
    "${TORCH_SPEC}"
# torchvision is only needed by the optional vLLM path; do not fail setup on it.
uv pip install --python "${VENV_PATH}/bin/python" \
    --index-url "https://download.pytorch.org/whl/${ROCM_WHEEL}" \
    torchvision || echo "WARNING: torchvision ROCm wheel not found; vLLM serving will be unavailable." >&2

echo "[3/4] Installing portable GSQ dependencies..."
uv pip install --python "${VENV_PATH}/bin/python" -r "${REPO_ROOT}/requirements-rocm.txt"
uv pip install --python "${VENV_PATH}/bin/python" -e "${REPO_ROOT}" --no-deps

if [[ "${INSTALL_VLLM}" = "1" ]]; then
    echo "[3b/4] Installing vLLM ROCm wheel (may pin its own torch)..."
    uv pip install --python "${VENV_PATH}/bin/python" \
        --extra-index-url https://wheels.vllm.ai/rocm/ --upgrade vllm
fi

echo "[4/4] Verifying..."
"${VENV_PATH}/bin/python" - <<'PYEOF'
import sys
import torch

print(f"torch      : {torch.__version__}")
print(f"hip        : {torch.version.hip}")
print(f"cuda build : {torch.version.cuda}")
print(f"available  : {torch.cuda.is_available()}")

if torch.version.hip is None:
    print("ERROR: this is not a ROCm build of PyTorch. Re-run the installer with the ROCm wheel index.", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("WARNING: ROCm build, but no GPU visible. Check 'rocminfo', groups (render/video), and HSA_OVERRIDE_GFX_VERSION.", file=sys.stderr)
else:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        arch = getattr(p, "gcnArchName", "?")
        print(f"  GPU {i}: {p.name}  arch={arch}  {p.total_memory/1024**3:.1f} GiB")
        if not str(arch).startswith(("gfx1200", "gfx1201", "gfx1150", "gfx1151", "gfx1152", "gfx1153")):
            print(f"  NOTE: port targeted gfx1200/gfx1201 and gfx115x; found {arch}.", file=sys.stderr)
        if str(arch).startswith("gfx115"):
            print(
                "  NOTE: RDNA3.5 APU: unified GTT memory, no dedicated VRAM. "
                "Keep the BIOS carve-out small; raise the TTM limit with "
                "'amd-ttm --set <GB>' if needed. gfx1152 needs ROCm >= 7.2.",
                file=sys.stderr,
            )

# Minimal bf16 GEMM (exercises hipBLAS/hipBLASLt on RDNA4).
a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print(f"bf16 matmul: ok (sum={float((a @ b).float().sum()):.2f})")

for pkg in ("transformers", "accelerate", "datasets", "compressed_tensors", "lion_pytorch"):
    try:
        m = __import__(pkg)
        print(f"  OK  {pkg} == {getattr(m, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"  FAIL {pkg}: {e}", file=sys.stderr)
        sys.exit(1)
PYEOF

echo "=========================================="
echo "Setup complete."
echo "  source ${VENV_PATH}/bin/activate"
echo "  # or: VENV_PATH=${VENV_PATH} bash scripts/run.sh"
echo "=========================================="
