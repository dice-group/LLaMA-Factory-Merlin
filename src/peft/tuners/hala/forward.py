from typing import Any, Optional

import torch

from peft.metrics import record_hala_metrics

from ..utils.language_routing import LANGUAGE_PAD_ID


def _apply_b_heads_packed(a_dot_x: torch.Tensor, b_list, route_weight: Optional[torch.Tensor]) -> torch.Tensor:
    if len(b_list) == 1:
        return b_list[0](a_dot_x)

    b_weight = torch.cat([b.weight for b in b_list], dim=0)
    out_all = torch.nn.functional.linear(a_dot_x.to(b_weight.dtype), b_weight, bias=None)
    out_all = out_all.view(a_dot_x.size(0), len(b_list), -1)
    if route_weight is None:
        return out_all.sum(dim=1)
    return (out_all * route_weight.to(out_all.dtype).unsqueeze(-1)).sum(dim=1)


def _can_use_grouped_mm(x: torch.Tensor) -> bool:
    if not x.is_cuda or not hasattr(torch.nn.functional, "grouped_mm"):
        return False
    try:
        return torch.cuda.get_device_capability(x.device) >= (8, 0)
    except RuntimeError:
        return False


def _pad_last_dim(x: torch.Tensor, size: int) -> torch.Tensor:
    pad = size - x.size(-1)
    if pad <= 0:
        return x
    return torch.nn.functional.pad(x, (0, pad))


def _align_grouped_mm_k(k: int) -> int:
    return ((int(k) + 7) // 8) * 8


def _expert_gate_from_topk(topi: torch.Tensor, weights: torch.Tensor, num_experts: int) -> torch.Tensor:
    gate = torch.zeros(topi.shape[:-1] + (num_experts,), device=weights.device, dtype=weights.dtype)
    return gate.scatter_add(-1, topi, weights)


def _apply_grouped_sparse_expert(
    layer,
    x: torch.Tensor,
    topi: torch.Tensor,
    weights: torch.Tensor,
    *,
    language_ids: Optional[torch.Tensor],
    expert_targets: Optional[torch.Tensor],
    result_dtype: torch.dtype,
    output_size: int,
) -> Optional[torch.Tensor]:
    if not _can_use_grouped_mm(x):
        return None

    batch, seq_len, _ = x.size()
    x_flat = x.reshape(-1, x.size(-1))
    topi_flat = topi.reshape(-1)
    weights_flat = weights.reshape(-1)
    token_idx = torch.arange(x_flat.size(0), device=x.device).repeat_interleave(topi.size(-1))

    order = torch.argsort(topi_flat, stable=True)
    experts = topi_flat[order]
    token_idx = token_idx[order]
    weights_sorted = weights_flat[order].to(result_dtype)
    counts = torch.bincount(experts, minlength=layer.num_experts).to(torch.int32)
    if int(counts.sum().item()) == 0:
        return torch.zeros((batch, seq_len, output_size), device=x.device, dtype=result_dtype)

    expert_names = [f"expert_{e}" for e in range(layer.num_experts)]
    b_counts = [len(layer.lora_B[name]) for name in expert_names]
    if len(set(b_counts)) != 1 or b_counts[0] <= 0:
        return None

    rank = int(layer.lora_A[expert_names[0]].out_features)
    if any(int(layer.lora_A[name].out_features) != rank for name in expert_names):
        return None

    padded_rank = _align_grouped_mm_k(rank)
    a_weights = []
    for name in expert_names:
        a_weights.append(_pad_last_dim(layer.lora_A[name].weight.transpose(0, 1), padded_rank).transpose(0, 1))
    a_weight = torch.stack(a_weights, dim=0).to(torch.bfloat16)
    a_weight = a_weight.transpose(1, 2)
    x_sel = x_flat[token_idx]
    dropped = layer.lora_dropout[expert_names[0]](x_sel.to(torch.bfloat16))
    offsets = counts.cumsum(0).to(torch.int32)
    a_out = torch.nn.functional.grouped_mm(dropped, a_weight, offs=offsets)

    num_heads = b_counts[0]
    out_features = int(layer.lora_B[expert_names[0]][0].out_features)
    b_weights = []
    for name in expert_names:
        packed = torch.cat([b.weight for b in layer.lora_B[name]], dim=0)
        b_weights.append(_pad_last_dim(packed, padded_rank))
    b_weight = torch.stack(b_weights, dim=0).to(torch.bfloat16).transpose(1, 2)
    b_out = torch.nn.functional.grouped_mm(a_out, b_weight, offs=offsets)

    if num_heads == 1:
        out_sorted = b_out
    else:
        route_chunks = []
        start = 0
        for e, count_tensor in enumerate(counts):
            count = int(count_tensor.item())
            if count == 0:
                continue
            end = start + count
            name = expert_names[e]
            lora_route = layer.lora_route[name] if name in layer.lora_route else None
            if lora_route is None:
                route_chunks.append(torch.full((count, num_heads), 1.0 / num_heads, device=x.device, dtype=b_out.dtype))
                start = end
                continue

            x_e = x_sel[start:end]
            route_dtype = lora_route.weight.dtype
            route_logits = lora_route(x_e.to(route_dtype)).to(x.dtype).unsqueeze(1)
            head_targets: Optional[torch.Tensor] = None
            language_ids_sel: Optional[torch.Tensor] = None
            use_head_guidance = layer.language_guidance_scope == "all"
            if use_head_guidance and language_ids is not None:
                batch_idx = token_idx[start:end] // seq_len
                language_ids_sel = language_ids[batch_idx]
                head_targets = layer._language_head_targets(language_ids_sel, name)
                if head_targets is not None and expert_targets is not None and torch.is_tensor(expert_targets):
                    expert_targets_sel = expert_targets[batch_idx]
                    mismatch = expert_targets_sel != int(e)
                    if mismatch.any():
                        head_targets = head_targets.clone()
                        head_targets[mismatch] = LANGUAGE_PAD_ID
                layer._cache_router_state(route_logits, language_ids_sel, f"hydra_head_{name}", head_targets)
            route_logits = layer._apply_language_bias_heads(route_logits, head_targets)
            route_weight = layer._head_router_weights(route_logits)
            route_weight = layer._enforce_language_heads(route_weight, head_targets).squeeze(1)
            route_chunks.append(route_weight.to(b_out.dtype))
            start = end

        route_weight_sorted = torch.cat(route_chunks, dim=0)
        b_out = b_out.view(b_out.size(0), num_heads, out_features)
        out_sorted = (b_out * route_weight_sorted.unsqueeze(-1)).sum(dim=1)

    scales = torch.tensor(
        [float(layer.scaling[name]) for name in expert_names],
        device=x.device,
        dtype=out_sorted.dtype,
    )
    scale_sorted = torch.repeat_interleave(scales, counts.to(torch.long), dim=0).unsqueeze(-1)
    out_sorted = out_sorted * scale_sorted * weights_sorted.to(out_sorted.dtype).unsqueeze(-1)

    moe_out_flat = torch.zeros((x_flat.size(0), output_size), device=x.device, dtype=result_dtype)
    moe_out_flat.index_add_(0, token_idx, out_sorted.to(result_dtype))
    return moe_out_flat.view(batch, seq_len, output_size)


def _apply_packed_dense_lowrank(
    layer,
    x: torch.Tensor,
    topi: torch.Tensor,
    weights: torch.Tensor,
    *,
    language_ids: Optional[torch.Tensor],
    expert_targets: Optional[torch.Tensor],
    result_dtype: torch.dtype,
) -> torch.Tensor:
    expert_names = [f"expert_{e}" for e in range(layer.num_experts)]
    first_a = layer.lora_A[expert_names[0]]
    rank = int(first_a.out_features)
    if any(int(layer.lora_A[name].out_features) != rank for name in expert_names):
        raise ValueError("packed_dense_lowrank requires all HALA experts to use the same rank.")

    x_cast = x.to(first_a.weight.dtype)
    dropped = layer.lora_dropout[expert_names[0]](x_cast)
    a_weight = torch.cat([layer.lora_A[name].weight for name in expert_names], dim=0)
    a_all = torch.nn.functional.linear(dropped, a_weight).view(x.size(0), x.size(1), layer.num_experts, rank)

    expert_gate = _expert_gate_from_topk(topi, weights, layer.num_experts).to(a_all.dtype)

    lowrank_chunks = []
    b_weights = []
    for e, name in enumerate(expert_names):
        b_list = layer.lora_B[name]
        if not b_list:
            continue

        expert_lowrank = a_all[:, :, e, :] * expert_gate[:, :, e].unsqueeze(-1)
        lora_route = layer.lora_route[name] if name in layer.lora_route else None
        if lora_route is None or len(b_list) == 1:
            lowrank_chunks.append(expert_lowrank * layer.scaling[name])
            b_weights.append(b_list[0].weight)
            continue

        route_dtype = lora_route.weight.dtype
        route_logits = lora_route(x.to(route_dtype)).to(x.dtype)
        head_targets: Optional[torch.Tensor] = None
        use_head_guidance = layer.language_guidance_scope == "all"
        if use_head_guidance and language_ids is not None:
            head_targets = layer._language_head_targets(language_ids, name)
            if head_targets is not None and expert_targets is not None and torch.is_tensor(expert_targets):
                mismatch = expert_targets != int(e)
                if mismatch.any():
                    head_targets = head_targets.clone()
                    head_targets[mismatch] = LANGUAGE_PAD_ID
            layer._cache_router_state(route_logits, language_ids, f"hydra_head_{name}", head_targets)

        route_logits = layer._apply_language_bias_heads(route_logits, head_targets)
        route_weight = layer._head_router_weights(route_logits)
        route_weight = layer._enforce_language_heads(route_weight, head_targets).to(expert_lowrank.dtype)
        for i, b in enumerate(b_list):
            lowrank_chunks.append(expert_lowrank * route_weight[:, :, i].unsqueeze(-1) * layer.scaling[name])
            b_weights.append(b.weight)

    packed_lowrank = torch.cat(lowrank_chunks, dim=-1)
    packed_b = torch.cat(b_weights, dim=1)
    return torch.nn.functional.linear(packed_lowrank.to(packed_b.dtype), packed_b, bias=None).to(result_dtype)


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

        x_cast = x.to(lora_A.weight.dtype)
        route_dtype = lora_route.weight.dtype
        route_logits = lora_route(x_cast.to(route_dtype)).to(result.dtype)
        use_head_guidance = layer.language_guidance_scope == "all"
        head_targets = layer._language_head_targets(language_ids, active_adapter) if use_head_guidance else None
        if use_head_guidance:
            layer._cache_router_state(route_logits, language_ids, f"hydra_head_{active_adapter}", head_targets)
        route_logits = layer._apply_language_bias_heads(route_logits, head_targets)
        route_weight = layer._head_router_weights(route_logits)
        route_weight = layer._enforce_language_heads(route_weight, head_targets)
        head_assign = torch.argmax(route_weight, dim=-1, keepdim=True)

        if layer.track_router_metrics:
            with torch.no_grad():
                token_count = route_weight.numel() // route_weight.size(-1)
                if token_count > 0:
                    flat_assign = head_assign.reshape(-1)
                    head_counts = torch.bincount(flat_assign, minlength=layer.lora_num[active_adapter]).to(torch.float32)
                    mean_head = head_counts.mean().item()
                    head_cv = float((head_counts.std(unbiased=False) / (mean_head + 1e-6)).item()) if mean_head > 0 else 0.0
                    head_active = float((head_counts > 0).float().mean().item())
                    head_entropy = float((-route_weight * torch.log(route_weight + 1e-8)).sum(dim=-1).mean().item())
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
                    record_hala_metrics(metrics, weight=metrics_weight)

        a_dot_x = lora_A(dropout(x_cast))
        if len(lora_B) == 1:
            result = result + lora_B[0](a_dot_x) * scaling
        else:
            b_weight = lora_B[0].weight
            if len(lora_B) > 1:
                b_weight = torch.cat([b.weight for b in lora_B], dim=0)
            out_all = torch.nn.functional.linear(a_dot_x.to(b_weight.dtype), b_weight, bias=None)
            out_all = out_all.view(a_dot_x.size(0), a_dot_x.size(1), len(lora_B), -1)
            out_weighted = (out_all * route_weight.to(out_all.dtype).unsqueeze(-1)).sum(dim=2)
            result = result + out_weighted.to(result.dtype) * scaling

    return result.to(torch_result_dtype)


def forward_expert(layer, x: torch.Tensor, *args: Any, language_ids: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
    result = layer.base_layer(x, *args, **kwargs)
    torch_result_dtype = result.dtype

    for shared_idx in range(int(getattr(layer, "num_shared_experts", 0) or 0)):
        shared_name = f"shared_expert_{shared_idx}"
        if shared_name not in layer.lora_A:
            continue
        result = result + layer._adapter_delta(x, shared_name, language_ids=None).to(result.dtype)

    router_dtype = getattr(layer.router.weight, "dtype", torch.float32)
    logits = layer.router(x.to(router_dtype)).to(x.dtype)

    use_expert_guidance = layer.language_guidance_scope in {"all", "expert_only"}
    expert_targets = layer._language_expert_targets(language_ids) if use_expert_guidance else None
    if use_expert_guidance:
        layer._cache_router_state(logits, language_ids, "hydra_expert", expert_targets)
    logits = layer._apply_language_bias_experts(logits, expert_targets)

    if layer.top_k == 1:
        topi = torch.argmax(logits, dim=-1, keepdim=True)
        router_probs = torch.softmax(logits.to(torch.float32), dim=-1).to(x.dtype)
        weights = router_probs.gather(-1, topi)
    else:
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
                load_cv = float((counts.std(unbiased=False) / (mean_load + 1e-6)).item()) if mean_load > 0 else 0.0
                router_probs = torch.softmax(logits.to(torch.float32), dim=-1)
                entropy = float((-router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean().item())
                weight_mean = float(weights.mean().item())
                metrics = {
                    "expert_load_cv": load_cv,
                    "expert_active_frac": active_frac,
                    "expert_router_entropy": entropy,
                    "expert_topk_weight_mean": weight_mean,
                    "expert_router_input_norm_mean": float(x.norm(dim=-1).mean().item()),
                    "expert_router_logit_std": float(logits.to(torch.float32).std(unbiased=False).item()),
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
                record_hala_metrics(metrics, weight=metrics_weight)

    execution_mode = getattr(layer, "hala_execution_mode", "dense_expert_dense_head")
    if execution_mode == "packed_dense_lowrank":
        moe_out = _apply_packed_dense_lowrank(
            layer,
            x,
            topi,
            weights,
            language_ids=language_ids,
            expert_targets=expert_targets,
            result_dtype=result.dtype,
        )
        result = result + moe_out
        return result.to(torch_result_dtype)

    if execution_mode == "grouped_sparse_expert_dense_head":
        moe_out = _apply_grouped_sparse_expert(
            layer,
            x,
            topi,
            weights,
            language_ids=language_ids,
            expert_targets=expert_targets,
            result_dtype=result.dtype,
            output_size=result.size(-1),
        )
        if moe_out is not None:
            result = result + moe_out
            return result.to(torch_result_dtype)

    use_sparse = execution_mode in {
        "sparse_expert_dense_head",
        "packed_sparse_expert_dense_head",
        "grouped_sparse_expert_dense_head",
    }
    use_sparse = use_sparse and layer.top_k < layer.num_experts
    use_packed_heads = execution_mode in {"packed_sparse_expert_dense_head", "grouped_sparse_expert_dense_head"}
    if use_sparse:
        batch, seq_len, _ = x.size()
        x_flat = x.reshape(-1, x.size(-1))
        topi_flat = topi.reshape(-1, topi.size(-1))
        weights_flat = weights.reshape(-1, weights.size(-1))

        moe_out_flat = torch.zeros((x_flat.size(0), result.size(-1)), device=result.device, dtype=result.dtype)

        for e in range(layer.num_experts):
            name = f"expert_{e}"
            b_list = layer.lora_B[name]
            if not b_list:
                continue

            mask = topi_flat == e
            if not mask.any():
                continue
            token_idx, kth = torch.where(mask)
            x_sel = x_flat[token_idx]

            route_weight_flat = None
            lora_route = layer.lora_route[name] if name in layer.lora_route else None
            use_head_router = lora_route is not None and len(b_list) > 1
            if use_head_router:
                route_dtype = lora_route.weight.dtype
                # Sparse expert mode should only pay head-router cost for tokens
                # that selected this expert. Keep a singleton sequence dimension so
                # the existing language-bias/head-enforcement helpers still apply.
                route_logits = lora_route(x_sel.to(route_dtype)).to(x.dtype).unsqueeze(1)
                head_targets: Optional[torch.Tensor] = None
                language_ids_sel: Optional[torch.Tensor] = None
                use_head_guidance = layer.language_guidance_scope == "all"
                if use_head_guidance and language_ids is not None:
                    batch_idx = token_idx // seq_len
                    language_ids_sel = language_ids[batch_idx]
                    head_targets = layer._language_head_targets(language_ids_sel, name)
                    if head_targets is not None and expert_targets is not None and torch.is_tensor(expert_targets):
                        expert_targets_sel = expert_targets[batch_idx]
                        mismatch = expert_targets_sel != int(e)
                        if mismatch.any():
                            head_targets = head_targets.clone()
                            head_targets[mismatch] = LANGUAGE_PAD_ID
                    layer._cache_router_state(route_logits, language_ids_sel, f"hydra_head_{name}", head_targets)
                route_logits = layer._apply_language_bias_heads(route_logits, head_targets)
                route_weight = layer._head_router_weights(route_logits)
                route_weight = layer._enforce_language_heads(route_weight, head_targets)
                route_weight_flat = route_weight.squeeze(1)

            a = layer.lora_A[name]
            drop = layer.lora_dropout[name]
            scale = layer.scaling[name]
            a_dot_x = a(drop(x_sel.to(a.weight.dtype)))

            if use_packed_heads:
                out = _apply_b_heads_packed(a_dot_x, b_list, route_weight_flat)
            elif route_weight_flat is None or len(b_list) == 1:
                out = sum(b(a_dot_x) for b in b_list)
            else:
                route_sel = route_weight_flat.to(a_dot_x.dtype)
                out = 0
                for i, b in enumerate(b_list):
                    out = out + b(a_dot_x) * route_sel[:, i].unsqueeze(-1)

            out = out * scale
            target_dtype = moe_out_flat.dtype
            out = out.to(target_dtype)
            weight_sel = weights_flat[token_idx, kth].to(target_dtype).unsqueeze(-1)
            moe_out_flat.index_add_(0, token_idx, out * weight_sel)

        moe_out = moe_out_flat.view(batch, seq_len, -1)
    else:
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
