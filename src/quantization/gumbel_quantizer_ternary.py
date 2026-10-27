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

class GumbelQuantizerTernary(nn.Module):
    def __init__(self, Q, scales, groupsize, std, strength, device, dtype, logits_dtype=None):
        super().__init__()
        self.weight_shape = tuple(Q.shape)
        self.device = device
        self.dtype = dtype
        self.logits_dtype = logits_dtype if logits_dtype is not None else dtype
        self.idx = torch.arange(self.weight_shape[1], device=self.device) // groupsize

        sign_logits = (std * (torch.randn_like(Q) + torch.sign(Q) * strength)).to(self.dtype)

        scale_per_col = scales[:, self.idx]
        Q = (Q / scale_per_col).abs()

        mask_logits = (std * (torch.randn_like(Q) + (2 * Q - 1) * strength)).to(self.dtype)

        self.sign_logits = nn.Parameter(sign_logits.to(self.logits_dtype).detach())
        self.mask_logits = nn.Parameter(mask_logits.to(self.logits_dtype).detach())
        self.scales = nn.Parameter(scales.float().detach())


    def forward(self, temperature, scale=1.0):
        return GumbelSoftmaxFunction.apply(
            self.sign_logits,
            self.mask_logits,
            self.scales,
            self.idx,
            float(temperature),
            float(scale),
            self.device,
            self.dtype
        )

    def get_hard_weights(self):
        hard_mask = (self.mask_logits > 0).to(self.dtype)
        hard_sign = torch.sign(self.sign_logits)
        scale_per_col = self.scales[:, self.idx].to(self.dtype)
        output = hard_mask * hard_sign * scale_per_col

        return output, self.scales.to(self.dtype)


class GumbelSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sign_logits, mask_logits, scales, idx, temperature, scale, device, dtype):
        ctx.save_for_backward(sign_logits, mask_logits, scales)
        ctx.idx = idx
        ctx.temperature = temperature
        ctx.scale = scale
        ctx.device = device
        ctx.dtype = dtype

        ctx.fwd_rng_state = _get_rng_state(device)

        eps = 1e-8
        u_mask = _rand_like(mask_logits)
        mask_noise = torch.logit(u_mask, eps=eps).to(ctx.dtype)
        soft_mask = torch.sigmoid((2.0 * mask_logits.to(ctx.dtype) * ctx.scale + mask_noise) / ctx.temperature)

        u_sign = _rand_like(sign_logits)
        sign_noise = torch.logit(u_sign, eps=eps).to(ctx.dtype)
        soft_sign = 2.0 * torch.sigmoid((2.0 * sign_logits.to(ctx.dtype) * ctx.scale + sign_noise) / ctx.temperature) - 1.0

        scale_per_col = scales[:, idx].to(ctx.dtype)

        output = soft_mask * soft_sign * scale_per_col

        return output

    @staticmethod
    def backward(ctx, grad_output):
        sign_logits, mask_logits, scales = ctx.saved_tensors
        idx = ctx.idx
        temperature = ctx.temperature
        scale = ctx.scale
        device = ctx.device
        dtype = ctx.dtype

        with _fork_rng(device):
            _set_rng_state(ctx.fwd_rng_state, device)
            eps = 1e-8
            u_mask = _rand_like(mask_logits)
            mask_noise = torch.logit(u_mask, eps=eps).to(dtype)
            soft_mask = torch.sigmoid((2.0 * mask_logits.to(dtype) * ctx.scale + mask_noise) / ctx.temperature)
    
            u_sign = _rand_like(sign_logits)
            sign_noise = torch.logit(u_sign, eps=eps).to(dtype)
            soft_sign = 2.0 * torch.sigmoid((2.0 * sign_logits.to(dtype) * ctx.scale + sign_noise) / ctx.temperature) - 1.0

        scale_per_col = scales[:, idx].to(dtype)

        grad_soft_mask = grad_output * soft_sign * scale_per_col
        grad_soft_sign = grad_output * soft_mask * scale_per_col
        grad_scale_per_col = grad_output * soft_mask * soft_sign

        grad_mask_logits = grad_soft_mask * soft_mask * (1.0 - soft_mask) * (2.0 * scale / temperature)

        grad_sign_logits = grad_soft_sign * (soft_sign + 1.0) * (1.0 - soft_sign) * scale / temperature

        grad_scales = torch.zeros_like(scales)
        idx_expanded = idx.unsqueeze(0).expand(grad_scales.size(0), -1)
        grad_scales.scatter_add_(1, idx_expanded, grad_scale_per_col.float())

        return grad_sign_logits.to(sign_logits.dtype), grad_mask_logits.to(mask_logits.dtype), grad_scales, None, None, None, None, None
