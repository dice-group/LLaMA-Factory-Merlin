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


def _needs_language_prior_cache(layer) -> bool:
    return float(getattr(layer, "language_prior_weight", 0.0) or 0.0) > 0.0


def _cache_expert_prior(layer, logits: torch.Tensor, language_ids: torch.Tensor, expert_targets: torch.Tensor) -> None:
    logits_for_loss = logits.mean(dim=1) if logits.dim() > 2 else logits
    layer._cache_router_state(logits_for_loss, language_ids, "hydra_expert", expert_targets)


def _cache_head_prior_for_selected_expert(
    layer,
    route_logits: torch.Tensor,
    topi_flat: torch.Tensor,
    batch_idx: torch.Tensor,
    expert_id: int,
    expert_name: str,
    language_ids: torch.Tensor,
    targets_by_batch: torch.Tensor,
) -> None:
    valid_token = (topi_flat == expert_id) & (targets_by_batch[batch_idx] >= 0)
    if not bool(valid_token.any().item()):
        return

    batch_size = int(language_ids.size(0))
    sums = torch.zeros((batch_size, route_logits.size(-1)), device=route_logits.device, dtype=route_logits.dtype)
    counts = torch.zeros((batch_size, 1), device=route_logits.device, dtype=route_logits.dtype)
    valid_batches = batch_idx[valid_token]
    sums.index_add_(0, valid_batches, route_logits[valid_token, expert_id])
    counts.index_add_(0, valid_batches, torch.ones((valid_batches.numel(), 1), device=route_logits.device, dtype=route_logits.dtype))

    has_tokens = counts.squeeze(-1) > 0
    layer._cache_router_state(
        sums[has_tokens] / counts[has_tokens].clamp_min(1),
        language_ids[has_tokens],
        f"hydra_head_{expert_name}",
        targets_by_batch[has_tokens],
    )


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
        route_weight = torch.ones((x_sel.size(0), 1), device=x_sel.device, dtype=x_sel.dtype)
        _record_head_metrics(
            layer,
            route_weight,
            None,
            None,
            expect_targets=False,
        )
        return route_weight

    lora_route = layer.lora_route[name] if name in layer.lora_route else None
    if lora_route is None:
        raise ValueError(f"HALA sparse-head routing requires a head router for {name}.")

    route_dtype = lora_route.weight.dtype
    route_logits = lora_route(x_sel.to(route_dtype)).to(x_sel.dtype).unsqueeze(1)
    batch_idx = token_idx // seq_len
    language_ids_sel = language_ids[batch_idx] if language_ids is not None and torch.is_tensor(language_ids) else None
    head_targets = None
    if language_ids_sel is not None:
        head_targets = _require_targets(layer._language_head_targets(language_ids_sel, name), "head")
        expert_targets_sel = _require_targets(expert_targets, "expert")[batch_idx]
        mismatch = expert_targets_sel != int(expert_id)
        if mismatch.any():
            head_targets = head_targets.clone()
            head_targets[mismatch] = LANGUAGE_PAD_ID
        if _needs_language_prior_cache(layer):
            layer._cache_router_state(route_logits.mean(dim=1), language_ids_sel, f"hydra_head_{name}", head_targets)

    route_logits = layer._apply_language_bias_heads(route_logits, head_targets)
    route_weight = layer._head_router_weights(route_logits)
    route_weight = layer._enforce_language_heads(route_weight, head_targets).squeeze(1)
    _record_head_metrics(
        layer,
        route_weight,
        head_targets,
        language_ids_sel,
        expect_targets=bool(head_targets is not None and (head_targets >= 0).any().item()),
    )
    return route_weight


def _packed_uniform_head_weights(
    layer,
    x_flat: torch.Tensor,
    topi_flat: torch.Tensor,
    weights_flat: torch.Tensor,
    seq_len: int,
    expert_names: list[str],
    language_ids: Optional[torch.Tensor],
    expert_targets: Optional[torch.Tensor],
) -> tuple[torch.Tensor, bool]:
    head_count = len(layer.lora_B[expert_names[0]])
    if any(len(layer.lora_B[name]) != head_count for name in expert_names):
        return torch.empty(0, device=x_flat.device), False
    if any(name not in layer.lora_route for name in expert_names):
        return torch.empty(0, device=x_flat.device), False

    route_dtype = layer.lora_route[expert_names[0]].weight.dtype
    route_weight = torch.cat([layer.lora_route[name].weight for name in expert_names], dim=0)
    route_logits = torch.nn.functional.linear(x_flat.to(route_dtype), route_weight)
    route_logits = route_logits.view(x_flat.size(0), layer.num_experts, head_count).to(x_flat.dtype)

    batch_idx = torch.arange(x_flat.size(0), device=x_flat.device) // seq_len
    language_ids_flat = language_ids[batch_idx] if language_ids is not None else None
    expert_targets_flat = _require_targets(expert_targets, "expert")[batch_idx] if expert_targets is not None else None

    head_targets = None
    if language_ids_flat is not None:
        per_expert_targets = []
        for expert_id, name in enumerate(expert_names):
            target = _require_targets(layer._language_head_targets(language_ids_flat, name), "head")
            if expert_targets_flat is not None:
                target = target.masked_fill(expert_targets_flat != int(expert_id), LANGUAGE_PAD_ID)
            per_expert_targets.append(target)
        head_targets = torch.stack(per_expert_targets, dim=1)

    if head_targets is not None and layer.language_head_router_mode == "bias":
        valid = head_targets >= 0
        bias = torch.zeros_like(route_logits)
        bias.scatter_(-1, head_targets.clamp_min(0).unsqueeze(-1), float(layer.language_head_bias_value or 0.0))
        route_logits = route_logits + bias * valid.unsqueeze(-1).to(bias.dtype)

    route_probs = layer._head_router_weights(route_logits)
    if head_targets is not None and layer.language_head_router_mode == "hard":
        valid = head_targets >= 0
        hard = torch.zeros_like(route_probs)
        hard.scatter_(-1, head_targets.clamp_min(0).unsqueeze(-1), 1)
        route_probs = torch.where(valid.unsqueeze(-1), hard, route_probs)

    expert_gate = torch.zeros((x_flat.size(0), layer.num_experts), device=x_flat.device, dtype=route_probs.dtype)
    expert_gate.scatter_(1, topi_flat.unsqueeze(1), weights_flat.to(route_probs.dtype).unsqueeze(1))
    scales = torch.tensor([float(layer.scaling[name]) for name in expert_names], device=x_flat.device, dtype=route_probs.dtype)

    needs_prior_cache = _needs_language_prior_cache(layer)
    if language_ids_flat is not None and head_targets is not None and (needs_prior_cache or layer.track_router_metrics):
        batch_size = int(language_ids.size(0)) if language_ids is not None else 0
        for expert_id, name in enumerate(expert_names):
            token_idx = torch.where(topi_flat == expert_id)[0]
            if token_idx.numel() == 0:
                continue
            if layer.track_router_metrics:
                if needs_prior_cache:
                    layer._cache_router_state(
                        route_logits[token_idx, expert_id].unsqueeze(1),
                        language_ids_flat[token_idx],
                        f"hydra_head_{name}",
                        head_targets[token_idx, expert_id],
                    )
                _record_head_metrics(
                    layer,
                    route_probs[token_idx, expert_id],
                    head_targets[token_idx, expert_id],
                    language_ids_flat[token_idx],
                    expect_targets=bool((head_targets[token_idx, expert_id] >= 0).any().item()),
                )
            elif needs_prior_cache and batch_size > 0:
                targets_by_batch = head_targets[torch.arange(batch_size, device=head_targets.device) * seq_len, expert_id]
                _cache_head_prior_for_selected_expert(
                    layer,
                    route_logits,
                    topi_flat,
                    batch_idx,
                    expert_id,
                    name,
                    language_ids,
                    targets_by_batch,
                )

    return route_probs * expert_gate.unsqueeze(-1) * scales.view(1, -1, 1), True


def forward_expert(layer, x: torch.Tensor, *args: Any, language_ids: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
    if layer.top_k != 1:
        raise ValueError("HALA exploration branch only supports sparse expert top_k=1.")
    if layer.head_top_k is not None and int(layer.head_top_k) not in (0, 1):
        raise ValueError("HALA exploration branch supports head_top_k=1 for sparse heads or 0 for dense heads.")
    if x.dim() != 3:
        raise ValueError(f"HALA sparse routing expects [batch, seq, hidden] input, got shape={tuple(x.shape)}.")
    if layer.language_guidance_scope != "all":
        raise ValueError("HALA requires language_guidance_scope='all' for expert and head LPR.")

    result = layer.base_layer(x, *args, **kwargs)
    result_dtype = result.dtype
    if layer.training or language_ids is not None or torch.is_tensor(language_ids):
        language_ids = _require_language_ids(language_ids, x.size(0))

    router_dtype = getattr(layer.router.weight, "dtype", torch.float32)
    logits = layer.router(x.to(router_dtype)).to(x.dtype)

    expert_targets = None
    if language_ids is not None:
        expert_targets = _require_targets(layer._language_expert_targets(language_ids), "expert")
        if _needs_language_prior_cache(layer):
            _cache_expert_prior(layer, logits, language_ids, expert_targets)
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
        expect_targets=expert_targets is not None,
    )

    batch, seq_len, _ = x.size()
    x_flat = x.reshape(-1, x.size(-1))
    topi_flat = topi.reshape(-1)
    weights_flat = weights.reshape(-1)
    expert_names = [f"expert_{idx}" for idx in range(layer.num_experts)]
    active_experts = [name for name in expert_names if name in layer.lora_A]
    dense_param_graph = False
    if active_experts:
        rank = int(layer.r[active_experts[0]])
        max_heads = max(len(layer.lora_B[name]) for name in active_experts)
        a_dtype = layer.lora_A[active_experts[0]].weight.dtype
        dropped = layer.lora_dropout[active_experts[0]](x_flat.to(a_dtype))
        a_weight = torch.cat([layer.lora_A[name].weight for name in expert_names], dim=0)
        a_all = torch.nn.functional.linear(dropped, a_weight).view(x_flat.size(0), layer.num_experts, rank)

        route_mask, dense_param_graph = _packed_uniform_head_weights(
            layer,
            x_flat,
            topi_flat,
            weights_flat,
            seq_len,
            expert_names,
            language_ids,
            expert_targets,
        )
        if not dense_param_graph:
            route_mask = torch.zeros(
                (x_flat.size(0), layer.num_experts, max_heads),
                device=x_flat.device,
                dtype=a_all.dtype,
            )
            for expert_id, name in enumerate(expert_names):
                token_idx = torch.where(topi_flat == expert_id)[0]
                if token_idx.numel() == 0:
                    continue
                route_weight = _head_weights_for_selected_tokens(
                    layer,
                    x_flat[token_idx],
                    name,
                    expert_id,
                    token_idx,
                    seq_len,
                    language_ids,
                    expert_targets,
                )
                scale = float(layer.scaling[name]) * weights_flat[token_idx].to(a_all.dtype)
                route_mask[token_idx, expert_id, : route_weight.size(-1)] = route_weight.to(a_all.dtype) * scale.unsqueeze(-1)

        b_columns = []
        for name in expert_names:
            b_list = layer.lora_B[name]
            for head_id in range(max_heads):
                if head_id < len(b_list):
                    b_columns.append(b_list[head_id].weight)
                else:
                    b_columns.append(torch.zeros_like(b_list[0].weight))
        b_weight = torch.cat(b_columns, dim=1)
        hidden = (a_all.unsqueeze(2) * route_mask.unsqueeze(-1)).reshape(x_flat.size(0), -1)
        moe_out_flat = torch.nn.functional.linear(hidden.to(b_weight.dtype), b_weight).to(result_dtype)
    else:
        moe_out_flat = torch.zeros((x_flat.size(0), result.size(-1)), device=result.device, dtype=result_dtype)

    result = result + moe_out_flat.view(batch, seq_len, -1)
    if not dense_param_graph:
        ddp_touch = _zero_touch_trainable_params(layer, dtype=result.dtype, device=result.device)
        if ddp_touch is not None:
            result = result + ddp_touch
    return result.to(result_dtype)
