from typing import Any, Optional
import torch
from peft.metrics import record_hydralora_metrics


def forward_flat(layer, x: torch.Tensor, *args: Any, language_ids: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
    result = layer.base_layer(x, *args, **kwargs)
    torch_result_dtype = result.dtype

    for active_adapter in layer._active_adapters:
        if active_adapter not in layer.lora_A:
            continue
        lora_A = layer.lora_A[active_adapter]
        lora_B = layer.lora_B[active_adapter]
        lora_route = layer.lora_route[active_adapter]

        dropout = layer.lora_dropout[active_adapter]
        scaling = layer.scaling[active_adapter]

        x = x.to(lora_A.weight.dtype)
        route_logits = lora_route(x.to(torch.float32)).to(result.dtype)
        use_head_guidance = layer.language_guidance_scope == "all"
        head_targets = layer._language_head_targets(language_ids, active_adapter) if use_head_guidance else None
        if use_head_guidance:
            layer._cache_router_state(route_logits, language_ids, f"hydra_head_{active_adapter}", head_targets)
        route_logits = layer._apply_language_bias_heads(route_logits, head_targets)
        route_weight = torch.softmax(route_logits, dim=-1, dtype=torch.float32).to(result.dtype)
        route_weight = layer._enforce_language_heads(route_weight, head_targets)
        head_assign = torch.argmax(route_weight, dim=-1, keepdim=True)

        if layer.track_router_metrics:
            with torch.no_grad():
                token_count = route_weight.numel() // route_weight.size(-1)
                if token_count > 0:
                    flat_assign = head_assign.reshape(-1)
                    head_counts = torch.bincount(
                        flat_assign,
                        minlength=layer.lora_num[active_adapter],
                    ).to(torch.float32)
                    mean_head = head_counts.mean().item()
                    if mean_head > 0:
                        head_cv = float((head_counts.std(unbiased=False) / (mean_head + 1e-6)).item())
                    else:
                        head_cv = 0.0
                    head_active = float((head_counts > 0).float().mean().item())
                    head_entropy = float(
                        (-route_weight * torch.log(route_weight + 1e-8)).sum(dim=-1).mean().item()
                    )
                    total_head_assign = head_counts.sum()
                    if total_head_assign > 0:
                        head_frac = head_counts / total_head_assign
                        head_max = float(head_frac.max().item())
                        head_min = float(head_frac.min().item())
                    else:
                        head_max = 0.0
                        head_min = 0.0
                    metrics = {
                        "head_load_cv": head_cv,
                        "head_active_frac": head_active,
                        "head_router_entropy": head_entropy,
                        "head_load_max_frac": head_max,
                        "head_load_min_frac": head_min,
                    }
                    metrics_weight = float(token_count)
                    metrics_weight = layer._append_target_metrics(
                        metrics=metrics,
                        metrics_weight=metrics_weight,
                        prefix="head",
                        target_tensor=head_targets,
                        selection=head_assign.squeeze(-1),
                        probs=route_weight,
                        language_ids=language_ids,
                        expect_targets=use_head_guidance and layer.language_list is not None,
                    )
                    record_hydralora_metrics(metrics, weight=metrics_weight)

        for i in range(layer.lora_num[active_adapter]):
            result = result + torch.unsqueeze(route_weight[:, :, i], -1) * lora_B[i](
                (lora_A(dropout(x)))
            ) * scaling

    return result.to(torch_result_dtype)


def forward_expert(layer, x: torch.Tensor, *args: Any, language_ids: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
    result = layer.base_layer(x, *args, **kwargs)
    torch_result_dtype = result.dtype

    router_dtype = getattr(layer.router.weight, "dtype", torch.float32)
    logits = layer.router(x.to(router_dtype)).to(x.dtype)

    use_expert_guidance = layer.language_guidance_scope in {"all", "expert_only"}
    expert_targets = layer._language_expert_targets(language_ids) if use_expert_guidance else None
    if use_expert_guidance:
        layer._cache_router_state(logits, language_ids, "hydra_expert", expert_targets)
    logits = layer._apply_language_bias_experts(logits, expert_targets)

    topv, topi = torch.topk(logits, layer.top_k, dim=-1)
    weights = torch.softmax(topv.to(torch.float32), dim=-1).to(x.dtype)
    topi, weights = layer._enforce_language_experts(topi, weights, expert_targets)

    if layer._should_debug_routing():
        layer._debug_routing_sample(x, language_ids, expert_targets, topi, weights)

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
                metrics = {
                    "expert_load_cv": load_cv,
                    "expert_active_frac": active_frac,
                    "expert_router_entropy": entropy,
                    "expert_topk_weight_mean": weight_mean,
                }
                total_assign = counts.sum()
                if total_assign > 0:
                    frac = counts / total_assign
                    metrics["expert_load_max_frac"] = float(frac.max().item())
                    metrics["expert_load_min_frac"] = float(frac.min().item())
                else:
                    metrics["expert_load_max_frac"] = 0.0
                    metrics["expert_load_min_frac"] = 0.0
                metrics_weight = float(token_count)
                metrics_weight = layer._append_target_metrics(
                    metrics=metrics,
                    metrics_weight=metrics_weight,
                    prefix="expert",
                    target_tensor=expert_targets,
                    selection=topi[:, :, 0],
                    probs=router_probs,
                    language_ids=language_ids,
                    expect_targets=use_expert_guidance and layer.language_list is not None,
                )
                record_hydralora_metrics(metrics, weight=metrics_weight)

    moe_out = torch.zeros_like(result, dtype=result.dtype)
    for e in range(layer.num_experts):
        expert_delta = layer._adapter_delta(
            x,
            f"expert_{e}",
            language_ids=language_ids,
            expert_id=e,
            expert_targets=expert_targets,
        ).to(moe_out.dtype)
        for k in range(layer.top_k):
            mask = topi[:, :, k].eq(e)
            if not mask.any():
                continue
            moe_out = moe_out + expert_delta * (weights[:, :, k] * mask.to(weights.dtype)).unsqueeze(-1)

    result = result + moe_out
    return result.to(torch_result_dtype)
