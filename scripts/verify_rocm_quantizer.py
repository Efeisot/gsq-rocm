#!/usr/bin/env python3
"""Isolate the riskiest ROCm path: the Gumbel-Softmax quantizer.

The custom autograd op captures the forward RNG state and replays it in
backward.  On ROCm this relies on HIP's RNG state save/restore and on bf16
softmax/logit kernels.  This script exercises exactly that, without loading a
model, so a failure here points directly at the quantizer rather than at the
model/attention stack.

Usage:
    python scripts/verify_rocm_quantizer.py
    GSQ_DEVICE=cpu python scripts/verify_rocm_quantizer.py    # CPU sanity check
"""

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import rocm_compat as rocm  # noqa: E402
from src.quantization import GumbelQuantizer2Bit  # noqa: E402


def main() -> int:
    device = rocm.normalize_device(os.environ.get("GSQ_DEVICE") or None)
    dtype = torch.bfloat16 if os.environ.get("GSQ_DTYPE", "bfloat16") == "bfloat16" else torch.float16
    print(f"[verify] {rocm.environment_summary()}")
    print(f"[verify] device={device} dtype={dtype}")

    torch.manual_seed(0)
    rows, cols, groupsize = 64, 512, 128
    n_groups = cols // groupsize

    W = torch.randn(rows, cols, device=device, dtype=torch.float32)
    scales = W.abs().amax(dim=1, keepdim=True).repeat(1, n_groups) / 2 + 1e-6
    idx = torch.arange(cols, device=device) // groupsize
    Q = W / scales[:, idx]

    quantizer = GumbelQuantizer2Bit(
        Q.to(dtype), scales, groupsize,
        std=0.01, strength=6.0,
        device=str(device), dtype=dtype, logits_dtype=torch.float32,
    )

    # 1) Determinism: same global seed => identical forward output.
    torch.manual_seed(1234)
    out_a = quantizer.forward(2.0, 100.0).float()
    torch.manual_seed(1234)
    out_b = quantizer.forward(2.0, 100.0).float()
    if not torch.equal(out_a, out_b):
        print("[verify] FAIL: forward is not deterministic under a fixed seed")
        return 1
    print("[verify] forward determinism: ok")

    # 2) Backward through the RNG-replay path, several steps.
    for step in range(3):
        torch.manual_seed(100 + step)
        out = quantizer.forward(2.0, 100.0 + step).float()
        loss = out.pow(2).mean()
        loss.backward()
        if quantizer.quant_logits.grad is None:
            print("[verify] FAIL: quant_logits.grad is None")
            return 1
        if not torch.isfinite(quantizer.quant_logits.grad).all():
            print(f"[verify] FAIL: non-finite gradient at step {step}")
            return 1
        print(f"[verify] step {step}: loss={loss.item():.6f} "
              f"grad_norm={quantizer.quant_logits.grad.norm().item():.6f}")
        quantizer.quant_logits.grad = None

    # 3) RNG helpers are usable on this device.
    state = rocm.get_rng_state(device)
    with rocm.fork_rng(device):
        _ = rocm.rand_like(quantizer.quant_logits)
    rocm.set_rng_state(state, device)
    print("[verify] rng state save/restore: ok")

    # 4) Hard weights.
    hard, hard_scales = quantizer.get_hard_weights()
    if not torch.isfinite(hard).all() or hard.shape != W.shape:
        print("[verify] FAIL: get_hard_weights() produced an invalid tensor")
        return 1
    print("[verify] hard weights: ok")

    print("[verify] ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
