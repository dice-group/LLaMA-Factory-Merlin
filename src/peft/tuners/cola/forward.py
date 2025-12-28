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

    use_sparse = layer.top_k < layer.num_experts
    if use_sparse:
        batch, seq_len, _ = x.size()
        x_flat = x.reshape(-1, x.size(-1))
        topi_flat = topi.reshape(-1, topi.size(-1))
        weights_flat = weights.reshape(-1, weights.size(-1))

        language_ids_flat = None
        expert_targets_flat = None
        if language_ids is not None and torch.is_tensor(language_ids):
            if language_ids.dim() == 1:
                language_ids_flat = language_ids.view(-1, 1).expand(-1, seq_len).reshape(-1)
            else:
                language_ids_flat = language_ids.reshape(-1)
        if language_targets is not None and torch.is_tensor(language_targets):
            if language_targets.dim() == 1:
                expert_targets_flat = language_targets.view(-1, 1).expand(-1, seq_len).reshape(-1)
            else:
                expert_targets_flat = language_targets.reshape(-1)

        moe_out_flat = torch.zeros(
            (x_flat.size(0), result.size(-1)),
            device=result.device,
            dtype=result.dtype,
        )
        for e in range(layer.num_experts):
            mask = topi_flat == e
            if not mask.any():
                continue
            token_idx, kth = torch.where(mask)
            x_sel = x_flat[token_idx]
            lang_sel = language_ids_flat[token_idx] if language_ids_flat is not None else None
            tgt_sel = expert_targets_flat[token_idx] if expert_targets_flat is not None else None
            expert_delta = layer._adapter_delta(
                x_sel,
                f"expert_{e}",
                language_ids=lang_sel,
                expert_id=e,
                expert_targets=tgt_sel,
            ).to(result.dtype)
            weight_sel = weights_flat[token_idx, kth].to(expert_delta.dtype).unsqueeze(-1)
            moe_out_flat.index_add_(0, token_idx, expert_delta * weight_sel)

        moe_out = moe_out_flat.view(batch, seq_len, -1)
    else:
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
