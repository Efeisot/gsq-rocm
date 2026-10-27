# GSQ on ROCm / AMD gfx120x (RDNA4) and gfx115x (RDNA3.5 APU)

This repository is a fork of [IST-DASLab/GSQ](https://github.com/IST-DASLab/GSQ)
patched for **AMD GPUs**:
discrete `gfx1200` / `gfx1201` (RX 9060 XT / RX 9070 family) and the RDNA3.5
Ryzen APUs `gfx1150` / `gfx1151` / `gfx1152` (Strix Point / Strix Halo /
Krackan Point).  The quantization algorithm is unchanged; only the CUDA-only
assumptions were replaced.

> The upstream (CUDA) tree is [IST-DASLab/GSQ](https://github.com/IST-DASLab/GSQ);
> its README is kept at [`UPSTREAM_README.md`](UPSTREAM_README.md).
> The crash points and fixes are documented in
> [`docs/PORTING_NOTES.tr.md`](docs/PORTING_NOTES.tr.md) (Turkish).
>
> `gfx1152` is an **APU**: the GPU has no dedicated VRAM and maps system RAM
> through GTT.  Use ROCm >= 7.2, keep the BIOS VRAM carve-out small, and see
> the APU section below before running anything large.

## What runs where

| Component | ROCm status |
|---|---|
| GPTQ init (`src/prior/gptq.py`) | **works** (fp32 Cholesky + hipBLAS) |
| Gumbel quantizers (1/2/3/4-bit, ternary) | **works** (RNG replay made device-agnostic) |
| Lion training loop (`src/trainer.py`) | **works** |
| Model wrappers (Llama/Qwen3/Qwen3-MoE/Qwen3.5/Gemma-4) | **works** with SDPA attention |
| Checkpoint writing (`compressed-tensors`) | **works** (CPU-side packing) |
| `save_model.py` reassembly | **works** |
| Humming / Marlin kernels | **not available** (CUDA-only; guarded with a clear error) |
| `flash-attn` | **not available** (GSQ already trains with SDPA) |
| vLLM serving | depends on the ROCm vLLM wheel; WNA16 fused-MoE Marlin is CUDA-only |

## Install

```bash
# gfx1200/gfx1201: ROCm >= 6.4.2 (7.0+ recommended)
# gfx115x APUs  : ROCm >= 7.2 on Linux (gfx1152 = Krackan Point)
bash scripts/setup_env_rocm.sh
source .venv-rocm/bin/activate
```

The installer auto-detects the gfx target with `rocminfo` and picks
`rocm7.2` for `gfx115x`, `rocm7.0` otherwise.  Override with
`ROCM_WHEEL=...` if your ROCm/PyTorch combination differs.  It deliberately
does **not** use `uv.lock` or `pyproject.toml`'s `[[tool.uv.index]]`: both pin
`torch+cu130` and `vllm==0.20.2` (CUDA wheels), which cannot see an AMD GPU.
It installs:

* `torch` / `torchvision` from `https://download.pytorch.org/whl/${ROCM_WHEEL}`;
* the portable deps from `requirements-rocm.txt`;
* GSQ itself with `--no-deps`.

Optional vLLM ROCm wheel:

```bash
INSTALL_VLLM=1 bash scripts/setup_env_rocm.sh
# or later:
uv pip install --extra-index-url https://wheels.vllm.ai/rocm/ --upgrade vllm
```

## Run a smoke test

```bash
export GSQ_RUNTIME="$PWD/runtime"

# 0) riskiest path only: RNG replay + bf16 softmax + custom autograd
python scripts/verify_rocm_quantizer.py

# Qwen3-0.6B, 2-bit, 2 layers (sanity check)
python main.py --config configs/rocm/gfx1200_qwen3_0.6b_2bit.yaml --max-layers 2

# full small run
bash scripts/run.sh --config configs/rocm/gfx1200_qwen3_0.6b_2bit.yaml

# reassemble
python save_model.py --config configs/rocm/gfx1200_qwen3_0.6b_2bit.yaml
```

`scripts/run.sh` detects ROCm through `/opt/rocm` / `rocminfo` / `hipcc` and
uses `.venv-rocm` when present.  Force the backend with `GSQ_BACKEND=rocm`.

### MiniCPM5-2B smoke test (CPU or gfx1152)

`configs/rocm/gfx1152_minicpm5_2b_2bit.yaml` and `..._3bit.yaml` are
deliberately tiny runs: MiniCPM5-2B (Llama architecture, 42 layers), one
layer, 2- or 3-bit, 4 calibration chunks of 128 tokens from the model-card
README (no dataset download).

```bash
# CPU-only validation of the port (no ROCm needed), 3-bit (2.5-3.5 band):
GSQ_DEVICE=cpu python main.py \
    --config configs/rocm/gfx1152_minicpm5_2b_3bit.yaml --max-layers 1

# 2-bit baseline for comparison:
GSQ_DEVICE=cpu python main.py \
    --config configs/rocm/gfx1152_minicpm5_2b_2bit.yaml --max-layers 1

# gfx1152 APU (same command without GSQ_DEVICE):
python main.py \
    --config configs/rocm/gfx1152_minicpm5_2b_3bit.yaml --max-layers 1
```

The corpus path is `data.local_text_path` (defaults to the model README);
`GSQ_LOCAL_TEXT=/path/to/file.txt` overrides it.  Perplexity evaluation is
disabled in these configs because the model wrappers call `.pin_memory()`,
which needs a CUDA/HIP accelerator.

CPU validation on this tree (1 layer, 4 x 128 tokens, 16-core host):

| bits | GPTQ avg loss | Gumbel val hard loss | layer time | MLP shard |
|---|---|---|---|---|
| 2 | 526.93 | 2.64e-04 | 70.0 s | 9.6 MiB |
| 3 | 131.49 | 6.99e-05 | 77.8 s | 15 MiB |

The 3-bit run stays inside the requested 2.5–3.5 bit band and reconstructs
the layer noticeably better than 2-bit at essentially the same CPU cost.  The
shard sizes also confirm the packing fix: 2-bit uses 16 values per int32 and
3-bit uses 10, instead of both being written as 4-bit containers.  These are
single-layer smoke numbers, not full-model perplexity.  MLP-only runs
(`self_attn: false`) leave attention in bf16, so they do not represent a full
3-bit model; use them to compare the quantizer, not final model size.

The same width is now propagated to `save_model.py`'s `quantization_config`
and to the checkpoint loader (which also infers the width from packed shapes
when resuming a shard), so a 3-bit shard is not silently read as 4-bit.

GPU validation on RX 9060 XT (gfx1200, ROCm 7.2.4, torch 2.10.0+rocm7.0):

| bits | GPTQ avg loss | Gumbel val hard loss | layer time |
|---|---|---|---|
| 2 | 528.05 | 2.60e-04 | 11.8 s |
| 3 | 131.67 | 6.91e-05 | 11.9 s |

The fair test (`configs/rocm/gfx1200_minicpm5_2b_3bit_fair.yaml`, 3-bit,
2 layers, 4 epochs) reduced val-hard MSE from 7.13e-5 to 6.45e-5 on layer 0
(-9.5%) and from 2.78e-3 to 2.48e-3 on layer 1 (-10.8%).  A fp32 Hessian
asymmetry that rocSOLVER rejected as non-positive-definite is handled by
`_safe_cholesky` in `src/prior/gptq.py`; `patches/gsq_safe_cholesky.patch`
applies the same fix to an unported tree.

## Environment variables added for ROCm

| Variable | Default | Meaning |
|---|---|---|
| `GSQ_BACKEND` | auto | `rocm` or `cuda`; selects allocator defaults and venv |
| `GSQ_DEVICE` | auto | `cuda` (ROCm namespace), `rocm`, `cpu` |
| `GSQ_ATTN_IMPLEMENTATION` | `sdpa` | `sdpa` or `eager`; `flash_attention_2` is rejected on ROCm |
| `GSQ_DIST_BACKEND` | `nccl` | PyTorch ROCm exposes RCCL as `nccl` |
| `GSQ_RNG_FP32` | `0` | set to `1` to sample Gumbel noise in fp32 and cast (fallback if a bf16 RNG kernel is missing) |
| `GSQ_ALLOW_HUMMING` | unset | bypass the Humming CUDA-only guard (only if you ported the kernels) |
| `PYTORCH_HIP_ALLOC_CONF` | `garbage_collection_threshold:0.8` | set by `scripts/_common.sh` |

Stability knobs (unset by default) are documented in
[`docs/PORTING_NOTES.tr.md`](docs/PORTING_NOTES.tr.md): `TORCH_BLAS_PREFER_HIPBLASLT=0`,
`HSA_OVERRIDE_GFX_VERSION=12.0.0`, `AMD_SERIALIZE_KERNEL=3`.

## gfx1152 / RDNA3.5 APU notes

`gfx1152` is **Krackan Point** (Ryzen AI 300-class APU, RDNA3.5).  Unlike
gfx120x it has **no dedicated VRAM**: GPU allocations are mapped from system
RAM through GTT/GPUVM.  Consequences:

* Use **ROCm >= 7.2** on Linux (ROCm 7.1.x is not supported for these APUs;
  7.2+ carries the required kernel fixes).  `gfx1151` additionally needs a
  recent kernel (ROCm documents >= 6.18.4 or Ubuntu HWE 6.17.0-19.19).
* The PyTorch installer auto-selects `rocm7.2` when `rocminfo` reports
  `gfx115x`.
* Keep the BIOS VRAM carve-out small and let GTT back the allocations.  GTT
  defaults to roughly half of RAM; raise the TTM limit if a model does not
  fit, e.g. `amd-ttm --set 32` (32 GiB), then reboot/reload the module.
* `torch.cuda.get_device_properties().total_memory` may report only the
  carve-out, not the full GTT budget.  Do not size runs from that number.
* GSQ's meta-init + one-layer-at-a-time streaming is the safest mode here:
  model weights are never all resident at once.  Keep `--max-layers` small
  for the first run and watch `free -g` / `amdgpu_top`.
* AMD's RDNA3.5 optimization guide recommends the usual APU tuning
  (`amd_iommu=off` or `iommu=pt`, GTT/TTM sizing) for best bandwidth.
* `HSA_OVERRIDE_GFX_VERSION=11.5.2` is only a last resort for older ROCm
  stacks that do not recognise gfx1152; it can silently use gfx1150 code
  objects.  Prefer a native ROCm 7.2+ install.

## Known limitations on a 16 GB gfx1200

* The paper-scale runs (Llama-70B, Kimi-K2.5) need far more VRAM than a
  consumer RDNA4 card.  Start with Qwen3-0.6B / Llama-3.2-1B (see `configs/rocm/`).
* `device_microbatch_size: 1` and `data.max_length <= 1024` keep activation
  buffers inside 16 GiB.
* `torch.cuda` is still the API namespace; `is_cuda` is `True` for HIP tensors.
* Use `HIP_VISIBLE_DEVICES` to restrict GPUs (PyTorch ROCm honours it; some
  releases also accept `CUDA_VISIBLE_DEVICES`, but do not rely on that).
* No `flash-attn`, no Humming, no Marlin.  Serving a GSQ checkpoint on ROCm
  currently means either (a) a ROCm vLLM build that can dequantize
  compressed-tensors, or (b) dequantize with `transformers` and serve bf16.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `HIP error: invalid device function` / `no kernel image` | The wheel was built without gfx1200/gfx1152. Use a matching ROCm PyTorch wheel (gfx1152: ROCm 7.2+). |
| Illegal instruction / garbled result in bf16 GEMM | `export TORCH_BLAS_PREFER_HIPBLASLT=0` (force rocBLAS). |
| GPU not visible | add user to `render`/`video`; check `rocminfo`; last resort `HSA_OVERRIDE_GFX_VERSION=12.0.0` (gfx120x) or `11.5.2` (gfx1152, only on old ROCm). |
| OOM / very slow on gfx1152 | APU GTT limit: shrink the BIOS carve-out, `amd-ttm --set <GB>`, and use `--max-layers 1` first. |
| Hang / crash in the allocator | keep `expandable_segments` off (default here). |
| Multi-GPU all-reduce hangs | RCCL works, but consumer RDNA4 has no xGMI; try `NCCL_P2P_DISABLE=1`. |
| vLLM refuses the checkpoint | WNA16 Marlin is CUDA-only; use the dequantized path. |

## Patches

`patches/` holds the two upstream-side fixes as standalone diffs, both already
applied in this tree (they are kept so the changes can be reviewed or ported
back to a plain CUDA checkout):

| Patch | Target | What it fixes |
|---|---|---|
| `gsq_safe_cholesky.patch` | `src/prior/gptq.py` | symmetrize + fp64 fallback when rocSOLVER rejects the fp32 Hessian as non-positive-definite |
| `gsq_num_bits_loader.patch` | `src/models/base.py` | packing width follows `gsq_bits` instead of the hard-coded 4 bits, so 2/3-bit shards are not written as 4-bit containers |

```bash
patch -p1 -d /path/to/GSQ < patches/gsq_num_bits_loader.patch
```

## Repository layout

The port was validated with this checkout layout:

```text
workspace/
├── gsq-rocm/       # this repository
├── rco-rocm/       # companion ROCm port of RCO
├── MiniCPM5-2B/    # optional: HF snapshot for the MiniCPM5 configs
└── llama.cpp/      # optional: patched tree for the RCO GGUF tools
```

`configs/rocm/gfx*_minicpm5_*.yaml` point at `../MiniCPM5-2B`, so a
`hf download openbmb/MiniCPM5-2B --local-dir ../MiniCPM5-2B` checkout is enough
for those examples.

**No model weights, checkpoints, GGUFs or logs are committed.** Every run
writes to `runtime/` (and the installer to `.venv-rocm/`); both are covered by
`.gitignore`, so `rm -rf runtime .venv-rocm` returns the tree to a clean
checkout.

## Verification

On ROCm 7.0 / PyTorch 2.10.0+rocm7.0, gfx1200 (RX 9060 XT, 16 GiB):

```bash
python scripts/verify_rocm_quantizer.py   # Gumbel replay + bf16 softmax + autograd
bash   scripts/verify_setup.sh            # environment / GPU sanity
```

`verify_rocm_quantizer.py` must end with `[verify] ALL OK`. The recorded run,
software versions and the single-layer numbers are in
[`docs/VERIFICATION.md`](docs/VERIFICATION.md).

## Attribution and license

GSQ is Apache-2.0, © the GSQ authors (IST-DASLab). This fork keeps the same
license; see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and the citation in
[`UPSTREAM_README.md`](UPSTREAM_README.md#citation).
