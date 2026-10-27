# Shared bootstrap for GSQ bare-metal entry scripts.
# Source from any scripts/*.sh after `set -euo pipefail`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "${REPO_ROOT}/.env"
    set +a
fi

# GSQ artifact root only — do not use machine-wide SCRATCH here (often set on HPC to a quota path).
GSQ_RUNTIME="${GSQ_RUNTIME:-${REPO_ROOT}/runtime}"
export GSQ_RUNTIME

# HuggingFace hub/datasets root. Override via `.env` or the shell when using a cluster cache.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
mkdir -p "${HF_HOME}/hub" "${HF_HOME}/datasets" "${HF_HOME}/xet" 2>/dev/null || true
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

export PYTHONUNBUFFERED=1

# ── Platform detection (CUDA vs ROCm) ──────────────────────────────────────
# GSQ_BACKEND=cuda|rocm overrides autodetection.
if [[ -z "${GSQ_BACKEND:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GSQ_BACKEND=cuda
    elif [[ -d /opt/rocm || -n "${ROCM_PATH:-}" ]] \
        || command -v rocminfo >/dev/null 2>&1 \
        || command -v hipcc >/dev/null 2>&1; then
        GSQ_BACKEND=rocm
    else
        GSQ_BACKEND=cuda
    fi
fi
export GSQ_BACKEND

# ROCm setup installs into .venv-rocm so CUDA and ROCm environments can coexist.
if [[ -z "${VENV_PATH:-}" ]]; then
    if [[ "${GSQ_BACKEND}" = "rocm" && -d "${REPO_ROOT}/.venv-rocm" ]]; then
        VENV_PATH="${REPO_ROOT}/.venv-rocm"
    else
        VENV_PATH="${REPO_ROOT}/.venv"
    fi
fi

if [[ -d "${VENV_PATH}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
else
    echo "WARNING: venv not found at ${VENV_PATH} — falling back to system python." >&2
    if [[ "${GSQ_BACKEND}" = "rocm" ]]; then
        echo "         Run 'bash scripts/setup_env_rocm.sh' to create the ROCm venv." >&2
    else
        echo "         Run 'bash scripts/setup_env.sh' to create one." >&2
    fi
fi

ulimit -c 0

if [[ "${GSQ_BACKEND}" = "rocm" ]]; then
    # gfx1200/gfx1201 (RDNA4) and gfx115x (RDNA3.5 APUs; gfx1152 needs ROCm
    # >= 7.2). expandable_segments is unsupported/unstable on several ROCm
    # releases, so default to the conservative allocator config. Both
    # spellings are honoured by PyTorch; HIP_* is the documented ROCm one.
    export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.8}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_HIP_ALLOC_CONF}}"
    # RCCL reads NCCL_* variables. Keep the CUDA-only async error handler out
    # of the ROCm path; it is a no-op at best and noisy at worst.
    export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-0}"
else
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
fi
# If a multi-GPU run hangs at the very end in destroy_process_group(), main.py bounds that
# call with GSQ_DIST_DESTROY_TIMEOUT_SEC (default 120 in code). Use 0 to wait indefinitely,
# or GSQ_SKIP_DIST_DESTROY=1 to skip destroy() entirely (exit may be noisier but returns).

mkdir -p "${REPO_ROOT}/logs"

detect_num_gpus() {
    if [[ "${GSQ_BACKEND}" = "rocm" ]] && command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi -L 2>/dev/null | grep -c '^GPU' || true
    elif command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l
    elif command -v python >/dev/null 2>&1; then
        python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null || echo 0
    else
        echo 0
    fi
}
