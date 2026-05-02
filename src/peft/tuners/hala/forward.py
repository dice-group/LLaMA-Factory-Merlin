from typing import Any, Optional

import torch

from peft.metrics import record_hala_metrics

from ..utils.language_routing import LANGUAGE_PAD_ID


def _zero_touch_trainable_params(layer, *, dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
    zero: Optional[torch.Tensor] = None

    modules = []
    router = getattr(layer, "router", None)
    if router is not None:
        modules.append(router)
    for name in ("lora_A", "lora_B", "lora_route"):
        store = getattr(layer, name, None)
        if store is None:
            continue
        modules.extend(store.values())

    for module in modules:
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            continue
        for param in parameters():
            if not param.requires_grad or param.numel() == 0:
                continue
            term = param.reshape(-1)[0].to(device=device, dtype=dtype)
            zero = term if zero is None else zero + term

    return None if zero is None else zero * 0.0


def _require_language_ids(language_ids: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
    if language_ids is None or not torch.is_tensor(language_ids):
        raise ValueError("HALA requires batch language_ids for expert and head routing.")
    if language_ids.dim() != 1 or language_ids.size(0) != batch_size:
        raise ValueError(
            f"HALA language_ids must have shape [{batch_size}], got {tuple(language_ids.shape)}."
        )
    if (language_ids < 0).any():
        raise ValueError("HALA requires non-pad language_ids for every sample in the batch.")
    return language_ids


def _require_targets(targets: Optional[torch.Tensor], stage: str) -> torch.Tensor:
    if targets is None or not torch.is_tensor(targets) or targets.numel() == 0:
        raise ValueError(f"HALA could not resolve {stage} targets from language_ids.")
    if (targets < 0).any():
        raise ValueError(f"HALA could not map every language_id to a {stage} target.")
    return targets


def _record_expert_metrics(
    layer,
    x: torch.Tensor,
    logits: torch.Tensor,
    topi: torch.Tensor,
    weights: torch.Tensor,
    expert_targets: Optional[torch.Tensor],
    language_ids: Optional[torch.Tensor],
    expect_targets: bool,
) -> None:
    if not layer.track_router_metrics:
        return

    with torch.no_grad():
        token_count = topi.numel()
        if token_count <= 0:
            return

        flat_indices = topi.reshape(-1)
        counts = torch.bincount(flat_indices, minlength=layer.num_experts).to(torch.float32)
        mean_load = counts.mean().item()
        total_assign = counts.sum()
        probs = torch.softmax(logits.to(torch.float32), dim=-1)
        metrics = {
            "expert_load_cv": float((counts.std(unbiased=False) / (mean_load + 1e-6)).item()) if mean_load > 0 else 0.0,
            "expert_active_frac": float((counts > 0).float().mean().item()),
            "expert_router_entropy": float((-probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()),
            "expert_topk_weight_mean": float(weights.mean().item()),
            "expert_router_input_norm_mean": float(x.norm(dim=-1).mean().item()),
            "expert_router_logit_std": float(logits.to(torch.float32).std(unbiased=False).item()),
        }
        if total_assign > 0:
            frac = counts / total_assign
            metrics["expert_load_max_frac"] = float(frac.max().item())
            metrics["expert_load_min_frac"] = float(frac.min().item())
        else:
            metrics["expert_load_max_frac"] = 0.0
            metrics["expert_load_min_frac"] = 0.0

        metrics_weight = layer._append_target_metrics(
            metrics=metrics,
            metrics_weight=float(token_count),
            prefix="expert",
            target_tensor=expert_targets,
            selection=topi[:, :, 0],
            probs=probs,
            language_ids=language_ids,
            expect_targets=expect_targets,
        )
        record_hala_metrics(metrics, weight=metrics_weight)


def _record_head_metrics(
    layer,
    route_weight: torch.Tensor,
    head_targets: Optional[torch.Tensor],
    language_ids: Optional[torch.Tensor],
    expect_targets: bool,
) -> None:
    if not layer.track_router_metrics or route_weight.numel() == 0:
        return

    with torch.no_grad():
        head_count = route_weight.size(-1)
        token_count = route_weight.numel() // head_count
        if token_count <= 0:
            return

        head_assign = torch.argmax(route_weight, dim=-1)
        counts = torch.bincount(head_assign.reshape(-1), minlength=head_count).to(torch.float32)
        mean_load = counts.mean().item()
        total_assign = counts.sum()
        metrics = {
            "head_load_cv": float((counts.std(unbiased=False) / (mean_load + 1e-6)).item()) if mean_load > 0 else 0.0,
            "head_active_frac": float((counts > 0).float().mean().item()),
            "head_router_entropy": float((-route_weight * torch.log(route_weight + 1e-8)).sum(dim=-1).mean().item()),
        }
        if total_assign > 0:
            frac = counts / total_assign
            metrics["head_load_max_frac"] = float(frac.max().item())
            metrics["head_load_min_frac"] = float(frac.min().item())
        else:
            metrics["head_load_max_frac"] = 0.0
            metrics["head_load_min_frac"] = 0.0

        metrics_weight = layer._append_target_metrics(
            metrics=metrics,
            metrics_weight=float(token_count),
            prefix="head",
            target_tensor=head_targets,
            selection=head_assign.unsqueeze(1),
            probs=route_weight.unsqueeze(1),
            language_ids=language_ids,
            expect_targets=expect_targets,
        )
        record_hala_metrics(metrics, weight=metrics_weight)


def _head_weights_for_selected_tokens(
    layer,
    x_sel: torch.Tensor,
    name: str,
    expert_id: int,
    token_idx: torch.Tensor,
    seq_len: int,
    language_ids: Optional[torch.Tensor],
    expert_targets: Optional[torch.Tensor],
) -> torch.Tensor:
    b_list = layer.lora_B[name]
    if len(b_list) <= 1:
        raise ValueError(f"HALA requires at least two routed heads for {name}.")

    lora_route = layer.lora_route[name] if name in layer.lora_route else None
    if lora_route is None:
        raise ValueError(f"HALA sparse-head routing requires a head router for {name}.")

    route_dtype = lora_route.weight.dtype
    route_logits = lora_route(x_sel.to(route_dtype)).to(x_sel.dtype).unsqueeze(1)
    batch_idx = token_idx // seq_len
    if language_ids is None or not torch.is_tensor(language_ids):
        raise ValueError("HALA requires batch language_ids for head routing.")
    language_ids_sel = language_ids[batch_idx]
    head_targets = _require_targets(layer._language_head_targets(language_ids_sel, name), "head")
    expert_targets_sel = _require_targets(expert_targets, "expert")[batch_idx]
    mismatch = expert_targets_sel != int(expert_id)
    if mismatch.any():
        head_targets = head_targets.clone()
        head_targets[mismatch] = LANGUAGE_PAD_ID

    layer._cache_router_state(route_logits, language_ids_sel, f"hydra_head_{name}", head_targets)

    route_logits = layer._apply_language_bias_heads(route_logits, head_targets)
    route_weight = layer._head_router_weights(route_logits)
    route_weight = layer._enforce_language_heads(route_weight, head_targets).squeeze(1)
    _record_head_metrics(
        layer,
        route_weight,
        head_targets,
        language_ids_sel,
        expect_targets=bool((head_targets >= 0).any().item()),
    )
    return route_weight


def forward_expert(layer, x: torch.Tensor, *args: Any, language_ids: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
    if layer.top_k != 1:
        raise ValueError("HALA exploration branch only supports sparse expert top_k=1.")
    if layer.head_top_k != 1:
        raise ValueError("HALA exploration branch only supports sparse head head_top_k=1.")
    if x.dim() != 3:
        raise ValueError(f"HALA sparse routing expects [batch, seq, hidden] input, got shape={tuple(x.shape)}.")
    if layer.language_guidance_scope != "all":
        raise ValueError("HALA requires language_guidance_scope='all' for expert and head LPR.")

    result = layer.base_layer(x, *args, **kwargs)
    result_dtype = result.dtype
    language_ids = _require_language_ids(language_ids, x.size(0))

    router_dtype = getattr(layer.router.weight, "dtype", torch.float32)
    logits = layer.router(x.to(router_dtype)).to(x.dtype)

    expert_targets = _require_targets(layer._language_expert_targets(language_ids), "expert")
    layer._cache_router_state(logits, language_ids, "hydra_expert", expert_targets)
    logits = layer._apply_language_bias_experts(logits, expert_targets)

    topv, topi = torch.topk(logits, 1, dim=-1)
    weights = torch.softmax(topv.to(torch.float32), dim=-1).to(x.dtype)
    topi, weights = layer._enforce_language_experts(topi, weights, expert_targets)

    if layer._should_debug_routing():
        layer._debug_routing_sample(x, language_ids, expert_targets, topi, weights)

    _record_expert_metrics(
        layer,
        x,
        logits,
        topi,
        weights,
        expert_targets,
        language_ids,
        expect_targets=True,
    )

    batch, seq_len, _ = x.size()
    x_flat = x.reshape(-1, x.size(-1))
    topi_flat = topi.reshape(-1)
    weights_flat = weights.reshape(-1)
    moe_out_flat = torch.zeros((x_flat.size(0), result.size(-1)), device=result.device, dtype=result_dtype)

    for expert_id in range(layer.num_experts):
        name = f"expert_{expert_id}"
        if name not in layer.lora_A:
            continue

        token_idx = torch.where(topi_flat == expert_id)[0]
        if token_idx.numel() == 0:
            continue

        x_sel = x_flat[token_idx]
        a = layer.lora_A[name]
        b_list = layer.lora_B[name]
        drop = layer.lora_dropout[name]
        scale = layer.scaling[name]

        a_dot_x = a(drop(x_sel.to(a.weight.dtype)))
        route_weight = _head_weights_for_selected_tokens(
            layer,
            x_sel,
            name,
            expert_id,
            token_idx,
            seq_len,
            language_ids,
            expert_targets,
        )

        out = torch.zeros((a_dot_x.size(0), result.size(-1)), device=result.device, dtype=result_dtype)
        head_idx = torch.argmax(route_weight, dim=-1)
        for head_id, b in enumerate(b_list):
            head_token_idx = torch.where(head_idx == head_id)[0]
            if head_token_idx.numel() == 0:
                continue
            out[head_token_idx] = b(a_dot_x[head_token_idx]).to(result_dtype)

        out = out.to(result_dtype) * float(scale) * weights_flat[token_idx].to(result_dtype).unsqueeze(-1)
        moe_out_flat.index_add_(0, token_idx, out)

    result = result + moe_out_flat.view(batch, seq_len, -1)
    ddp_touch = _zero_touch_trainable_params(layer, dtype=result.dtype, device=result.device)
    if ddp_touch is not None:
        result = result + ddp_touch
    return result.to(result_dtype)
