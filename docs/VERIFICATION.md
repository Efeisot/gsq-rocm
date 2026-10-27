# Verification — GSQ ROCm port

Recorded on the machine used for the port. Re-run the commands below to check a
fresh checkout.

## Environment

| Item | Value |
|---|---|
| GPU | AMD Radeon RX 9060 XT, `gfx1200`, 16 GiB (15.9 GiB visible) |
| ROCm | 7.0 (`7.0.51831` via `torch.version.hip`) |
| PyTorch | `2.10.0+rocm7.0` |
| Python | 3.12 |
| Host | Ryzen 9 7950X, 64 GB RAM, Linux |

```bash
$ rocminfo | grep -E "Marketing Name|gfx"
  Marketing Name:          AMD Radeon RX 9060 XT
      Name:                amdgcn-amd-amdhsa--gfx1200
```

## GPU quantizer check

Covers the riskiest ROCm paths: Gumbel RNG replay, bf16 softmax, the custom
autograd op and a three-step training loop.

```bash
$ python scripts/verify_rocm_quantizer.py
[verify] torch=2.10.0+rocm7.0 backend=ROCm 7.0.51831 devices=[0] AMD Radeon Graphics (gfx1200, 15.9 GiB)
[verify] device=cuda dtype=torch.bfloat16
[verify] forward determinism: ok
[verify] step 0: loss=0.470172 grad_norm=0.177129
[verify] step 1: loss=0.471970 grad_norm=0.179148
[verify] step 2: loss=0.475172 grad_norm=0.180750
[verify] rng state save/restore: ok
[verify] hard weights: ok
[verify] ALL OK
```

## Environment check

```bash
$ bash scripts/verify_setup.sh
==========================================
GSQ verify_setup
Visible GPUs : 1
==========================================
Python: 3.12.x ...
torch:  2.10.0+rocm7.0
ROCm build: True (hip=7.0.51831)
CUDA available: True
  [0] AMD Radeon Graphics  arch=gfx1200  15.9 GiB
GPU smoke matmul ok: 2048.0
Single-GPU verification complete.
```

## End-to-end runs

The full MiniCPM5-2B checks (one layer, 2/3-bit, GPTQ + Gumbel training, shard
write/read) and the fair 2-layer comparison are in the main
[README](../README.md#run-a-smoke-test). Summary:

| run | hardware | result |
|---|---|---|
| 1 layer, 2-bit, 4 x 128 tokens | CPU | GPTQ loss 526.93, val hard 2.64e-04, 70.0 s |
| 1 layer, 3-bit, 4 x 128 tokens | CPU | GPTQ loss 131.49, val hard 6.99e-05, 77.8 s |
| 1 layer, 2-bit, 4 x 128 tokens | gfx1200 | GPTQ loss 528.05, val hard 2.60e-04, 11.8 s |
| 1 layer, 3-bit, 4 x 128 tokens | gfx1200 | GPTQ loss 131.67, val hard 6.91e-05, 11.9 s |
| 2 layers, 3-bit, 4 epochs (fair test) | gfx1200 | val hard 7.13e-5 -> 6.45e-5 (layer 0), 2.78e-3 -> 2.48e-3 (layer 1) |

These are single-layer/partial-model checks; they validate the port, not final
model quality.
