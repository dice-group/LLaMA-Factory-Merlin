from typing import Any, Optional
import torch
from peft.metrics import record_cola_metrics


def forward_flat(layer, x: torch.Tensor, *args: Any, language_ids: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
    result = layer.base_layer(x, *args, **kwargs)
    torch_result_dtype = result.dtype
    for active_adapter in layer._active_adapters:
        if active_adapter not in layer.lora_A:
            continue
        result = result + layer._adapter_delta(x, active_adapter, language_ids=language_ids)
    return result.to(torch_result_dtype)


def forward_expert(layer, x: torch.Tensor, *args: Any, language_ids: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
    result = layer.base_layer(x, *args, **kwargs)
    torch_result_dtype = result.dtype

    router_dtype = layer.router.weight.dtype
    router_inp = x.to(router_dtype)
    logits = layer.router(router_inp)
    language_targets = layer._language_expert_targets(language_ids)
    if layer.language_guidance_scope != "none":
        layer._cache_router_state(logits, language_ids, language_targets)
    logits = layer._apply_language_bias(logits, language_targets)
    topv, topi = torch.topk(logits, layer.top_k, dim=-1)
    weights = torch.softmax(topv.to(torch.float32), dim=-1).to(x.dtype)
    topi, weights = layer._enforce_language_routing(topi, weights, language_targets)

    if layer._should_debug_routing():
        layer._debug_routing_sample(x, language_ids, language_targets, topi, weights)

    if layer.track_router_metrics:
        with torch.no_grad():
            token_count = topi.numel()
            if token_count > 0:
                flat_indices = topi.reshape(-1)
                counts = torch.bincount(flat_indices, minlength=layer.num_experts).to(torch.float32)
                active_frac = float((counts > 0).float().mean().item())
                mean_load = counts.mean().item()
                if mean_load > 0:
                    load_cv = float((counts.std(unbiased=False) / (mean_load + 1e-6)).item())
                else:
                    load_cv = 0.0

                router_probs = torch.softmax(logits.to(torch.float32), dim=-1)
                entropy = float((-router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean().item())
                weight_mean = float(weights.mean().item())

                total_assign = counts.sum()
                if total_assign > 0:
                    load_frac = counts / total_assign
                    max_frac = float(load_frac.max().item())
                    min_frac = float(load_frac.min().item())
                else:
                    max_frac = 0.0
                    min_frac = 0.0

                metrics = {
                    "expert_load_cv": load_cv,
                    "active_expert_frac": active_frac,
                    "router_entropy": entropy,
                    "topk_weight_mean": weight_mean,
                    "expert_load_max_frac": max_frac,
                    "expert_load_min_frac": min_frac,
                }
                metrics_weight = float(token_count)
                metrics_weight = layer._append_language_target_metrics(
                    metrics=metrics,
                    metrics_weight=metrics_weight,
                    language_targets=language_targets,
                    language_ids=language_ids,
                    router_probs=router_probs,
                    top_indices=topi,
                )
                record_cola_metrics(metrics, weight=metrics_weight)

    expert_outs = [
        layer._adapter_delta(
            x,
            f"expert_{e}",
            language_ids=language_ids,
            expert_id=e,
            expert_targets=language_targets,
        )
        for e in range(layer.num_experts)
    ]
    expert_outs = torch.stack(expert_outs, dim=-1)  # [B, S, D_out, E]

    topi_expanded = topi.unsqueeze(2).expand(-1, -1, expert_outs.size(2), -1)
    gathered = torch.gather(expert_outs, dim=3, index=topi_expanded)
    weights_expanded = weights.unsqueeze(2)
    moe_out = (gathered * weights_expanded).sum(dim=-1)

    result = result + moe_out
    return result.to(torch_result_dtype)
