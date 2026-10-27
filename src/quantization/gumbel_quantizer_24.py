import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # package import (normal entry points)
    from ..rocm_compat import (
        fork_rng as _fork_rng,
        get_rng_state as _get_rng_state,
        rand_like as _rand_like,
        set_rng_state as _set_rng_state,
    )
except ImportError:  # direct/standalone import
    from rocm_compat import (  # type: ignore
        fork_rng as _fork_rng,
        get_rng_state as _get_rng_state,
        rand_like as _rand_like,
        set_rng_state as _set_rng_state,
    )

class GumbelQuantizer24(nn.Module):
    def __init__(self, W, mask, std, strength, device, dtype, logits_dtype=None):
        super().__init__()
        self.weight_shape = tuple(W.shape)
        self.device = device
        self.dtype = dtype
        self.register_buffer("possible_masks", torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=dtype, device=self.device))

        logits = torch.matmul(mask.reshape(-1, 4), self.possible_masks.T)
        logits = logits - logits.mean(dim=0, keepdim=True)
        mask_logits = std * (torch.randn_like(logits) + logits * strength)

        self.W = nn.Parameter(W.to(device=self.device, dtype=logits_dtype).detach().clone())
        self.mask_logits = nn.Parameter(mask_logits.to(logits_dtype).detach())

    def forward(self, temperature, scale=1.0):
        return GumbelSoftmaxFunction.apply(
            self.mask_logits,
            self.W,
            self.possible_masks,
            float(temperature),
            float(scale),
            self.device
        )

    def get_hard_weights(self):
        hard_mask_idx = torch.argmax(self.mask_logits, dim=-1)
        output = self.possible_masks[hard_mask_idx].view_as(self.W) * self.W.to(self.dtype)

        return output, self.W.to(self.dtype)

class GumbelSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_logits, W, possible_masks, temperature, scale, device):
        ctx.save_for_backward(mask_logits, W)
        ctx.possible_masks = possible_masks
        ctx.temperature = temperature
        ctx.scale = scale
        ctx.device = device

        ctx.fwd_rng_state = _get_rng_state(device)

        eps = 1e-8

        u = _rand_like(mask_logits)
        noise = -torch.log(-torch.log(u + eps) + eps)
        soft_mask = F.softmax((mask_logits * ctx.scale + noise) / ctx.temperature, dim=-1).to(possible_masks.dtype)
        soft_output = torch.matmul(soft_mask, possible_masks)

        output = soft_output.view_as(W) * W.to(possible_masks.dtype)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask_logits, W = ctx.saved_tensors
        possible_masks = ctx.possible_masks
        temperature = ctx.temperature
        scale = ctx.scale
        device = ctx.device

        with _fork_rng(device):
            _set_rng_state(ctx.fwd_rng_state, device)
            eps = 1e-8
            u = _rand_like(mask_logits)
            noise = -torch.log(-torch.log(u + eps) + eps)
            soft_mask = F.softmax((mask_logits * ctx.scale + noise) / ctx.temperature, dim=-1).to(possible_masks.dtype)

        soft_output = torch.matmul(soft_mask, possible_masks)
        grad_W = grad_output * soft_output

        grad_out_flat = (grad_output * W).reshape(-1, 4)

        grad_soft_mask = torch.matmul(grad_out_flat, possible_masks.T)

        dot = (grad_soft_mask * soft_mask).sum(dim=-1, keepdim=True)
        grad_z = soft_mask * (grad_soft_mask - dot)

        grad_mask_logits = grad_z * (scale / temperature)

        return grad_mask_logits.to(mask_logits.dtype), grad_W.to(W.dtype), None, None, None, None
