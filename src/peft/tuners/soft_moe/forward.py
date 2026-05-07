from typing import Any, Optional

import torch
import torch.nn.functional as F

from peft.metrics import record_soft_moe_metrics

from ..utils.language_routing import LANGUAGE_PAD_ID


def _cache_expert_prior(layer, logits: torch.Tensor, language_ids: torch.Tensor, expert_targets: torch.Tensor) -> None:
    logits_for_loss = logits.mean(dim=1) if logits.dim() > 2 else logits
    layer._cache_router_state(logits_for_loss, language_ids, "soft_moe_expert", expert_targets)


def _record_soft_moe_metrics(
    layer,
    logits: torch.Tensor,
    weights: torch.Tensor,
    expert_targets: Optional[torch.Tensor],
    language_ids: Optional[torch.Tensor],
) -> None:
    if not layer.track_router_metrics:
        return

    with torch.no_grad():
        probs = weights.to(torch.float32)
        token_count = probs.shape[0] * probs.shape[1] if probs.dim() == 3 else probs.shape[0]
        if token_count <= 0:
            return

        probs_flat = probs.reshape(-1, probs.size(-1))
        mean_weight = probs_flat.mean(dim=0)
        mean_load = mean_weight.mean().item()

        metrics = {
            "expert_load_cv": float((mean_weight.std(unbiased=False) / (mean_load + 1e-6)).item()),
            "expert_router_entropy": float((-probs_flat * torch.log(probs_flat + 1e-8)).sum(dim=-1).mean().item()),
            "expert_weight_max_mean": float(probs_flat.max(dim=-1).values.mean().item()),
            "expert_weight_min_mean": float(probs_flat.min(dim=-1).values.mean().item()),
        }

        if expert_targets is not None and language_ids is not None:
            targets_expanded = expert_targets.unsqueeze(1).expand(-1, probs.size(1)).reshape(-1)
            valid = targets_expanded >= 0
            if valid.any():
                target_probs = probs_flat[valid].gather(1, targets_expanded[valid].unsqueeze(1)).squeeze(1)
                metrics["expert_target_prob_mean"] = float(target_probs.mean().item())
                predicted = probs_flat[valid].argmax(dim=-1)
                metrics["expert_target_accuracy"] = float((predicted == targets_expanded[valid]).float().mean().item())

        record_soft_moe_metrics(metrics, weight=float(token_count))


def forward_soft_moe(layer, x: torch.Tensor, *args: Any, language_ids: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"SoftMoE expects [batch, seq, hidden] input, got shape={tuple(x.shape)}.")

    result = layer.base_layer(x, *args, **kwargs)
    result_dtype = result.dtype

    if language_ids is not None and torch.is_tensor(language_ids):
        if language_ids.dim() != 1 or language_ids.size(0) != x.size(0):
            raise ValueError(f"language_ids must have shape [{x.size(0)}], got {tuple(language_ids.shape)}.")
    elif layer.training:
        raise ValueError("SoftMoE training requires language_ids.")

    router_dtype = getattr(layer.router.weight, "dtype", torch.float32)
    logits = layer.router(x.to(router_dtype)).to(x.dtype)
    temperature = getattr(layer, "soft_moe_temperature", 1.0)
    weights = F.softmax(logits / temperature, dim=-1)

    expert_targets = None
    if language_ids is not None:
        expert_targets = layer._language_expert_targets(language_ids)
        if float(getattr(layer, "language_prior_weight", 0.0) or 0.0) > 0.0:
            _cache_expert_prior(layer, logits, language_ids, expert_targets)

    _record_soft_moe_metrics(layer, logits, weights, expert_targets, language_ids)

    batch, seq_len, hidden = x.size()
    expert_names = [f"expert_{idx}" for idx in range(layer.num_experts)]
    active_experts = [name for name in expert_names if name in layer.lora_A]

    if not active_experts:
        return result

    rank = int(layer.r[active_experts[0]])
    a_dtype = layer.lora_A[active_experts[0]].weight.dtype
    x_flat = x.reshape(-1, hidden)
    dropped = layer.lora_dropout[active_experts[0]](x_flat.to(a_dtype))

    a_weight = torch.cat([layer.lora_A[name].weight for name in expert_names], dim=0)
    a_all = F.linear(dropped, a_weight).view(x_flat.size(0), layer.num_experts, rank)

    weights_flat = weights.reshape(x_flat.size(0), layer.num_experts)
    scales = torch.tensor([float(layer.scaling[name]) for name in expert_names], device=x.device, dtype=a_all.dtype)
    weighted_a = (a_all * (weights_flat.unsqueeze(-1) * scales.view(1, -1, 1))).reshape(x_flat.size(0), -1)

    b_columns = []
    for name in expert_names:
        b_list = layer.lora_B[name]
        b_columns.append(b_list[0].weight)
    b_weight = torch.cat(b_columns, dim=1)

    moe_out_flat = F.linear(weighted_a.to(b_weight.dtype), b_weight).to(result_dtype)
    result = result + moe_out_flat.view(batch, seq_len, -1)
    return result
