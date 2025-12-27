import torch
from typing import Callable, Optional

LANGUAGE_PAD_ID = -1


def language_expert_targets(
    language_ids: Optional[torch.Tensor],
    mapping: Optional[torch.Tensor],
    guidance_scope: str,
    *,
    pad_id: int = LANGUAGE_PAD_ID,
) -> Optional[torch.Tensor]:
    if (
        language_ids is None
        or mapping is None
        or not torch.is_tensor(language_ids)
        or guidance_scope == "none"
        or mapping.numel() == 0
    ):
        return None
    mapping = mapping.to(language_ids.device)
    expert_ids = torch.full_like(language_ids, pad_id)
    valid = (language_ids >= 0) & (language_ids < mapping.numel())
    if valid.any():
        expert_ids[valid] = mapping[language_ids[valid]]
    return expert_ids


def language_head_targets(
    language_ids: Optional[torch.Tensor],
    language_list: Optional[list[str]],
    mapping: Optional[torch.Tensor],
    head_count: Optional[int],
    guidance_scope: str,
    *,
    pad_id: int = LANGUAGE_PAD_ID,
) -> Optional[torch.Tensor]:
    if (
        language_ids is None
        or language_list is None
        or not torch.is_tensor(language_ids)
        or guidance_scope != "all"
        or not head_count
    ):
        return None
    if mapping is not None and torch.is_tensor(mapping) and mapping.numel() > 0:
        mapping = mapping.to(language_ids.device)
        head_ids = torch.full_like(language_ids, pad_id)
        valid = (language_ids >= 0) & (language_ids < mapping.numel())
        if valid.any():
            candidate = mapping[language_ids[valid]]
            candidate = torch.where(candidate < head_count, candidate, torch.full_like(candidate, pad_id))
            head_ids[valid] = candidate
        return head_ids

    head_ids = torch.full_like(language_ids, pad_id)
    valid = (language_ids >= 0) & (language_ids < len(language_list))
    if valid.any():
        head_ids[valid] = language_ids[valid] % head_count
    return head_ids


def apply_language_bias(
    logits: torch.Tensor,
    target_ids: Optional[torch.Tensor],
    router_mode: str,
    bias_value: float,
) -> torch.Tensor:
    if target_ids is None or router_mode != "bias":
        return logits
    valid = target_ids >= 0
    if not valid.any():
        return logits
    bias = torch.zeros(logits.size(0), logits.size(-1), device=logits.device, dtype=logits.dtype)
    bias[valid, target_ids[valid]] = bias_value
    return logits + bias.unsqueeze(1)


def enforce_language_expert_routing(
    topi: torch.Tensor,
    weights: torch.Tensor,
    expert_ids: Optional[torch.Tensor],
    router_mode: str,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if expert_ids is None or router_mode != "hard":
        return topi, weights
    valid = expert_ids >= 0
    if not valid.any():
        return topi, weights
    seq_len = topi.size(1)
    replacement = expert_ids[valid].view(-1, 1, 1).expand(-1, seq_len, top_k)
    topi = topi.clone()
    weights = weights.clone()
    topi[valid] = replacement
    weights[valid] = 0
    weights[valid, :, 0] = 1
    return topi, weights


def enforce_language_head_weights(
    weights: torch.Tensor,
    head_ids: Optional[torch.Tensor],
    router_mode: str,
) -> torch.Tensor:
    if head_ids is None or router_mode != "hard":
        return weights
    valid = head_ids >= 0
    if not valid.any():
        return weights
    weights = weights.clone()
    weights[valid] = 0
    weights[valid, :, head_ids[valid]] = 1
    return weights


def append_router_target_metrics(
    metrics: dict[str, float],
    metrics_weight: float,
    *,
    prefix: str,
    target_tensor: Optional[torch.Tensor],
    selection: torch.Tensor,
    probs: torch.Tensor,
    language_ids: Optional[torch.Tensor],
    expect_targets: bool,
    on_missing: Optional[Callable[[str], None]] = None,
) -> float:
    key_prefix = f"{prefix}_target_"
    if target_tensor is not None and torch.is_tensor(target_tensor):
        valid_batch = target_tensor >= 0
        seq_len = probs.size(1)
        if valid_batch.any():
            expanded_targets = target_tensor[valid_batch].view(-1, 1).expand(-1, seq_len)
            top1 = selection[valid_batch]
            target_hits = (top1 == expanded_targets).float()
            target_probs = probs[valid_batch].gather(
                -1,
                target_tensor[valid_batch].view(-1, 1, 1).expand(-1, seq_len, 1),
            ).squeeze(-1)
            valid_tokens = target_probs.numel()
            target_entropy = float((-torch.log(target_probs + 1e-8)).mean().item())
            metrics.update(
                {
                    f"{key_prefix}hit_rate": float(target_hits.mean().item()),
                    f"{key_prefix}prob_mean": float(target_probs.mean().item()),
                    f"{key_prefix}neglogp": target_entropy,
                    f"{key_prefix}token_frac": float(valid_tokens / max(seq_len * valid_batch.sum().item(), 1)),
                }
            )
            return float(valid_tokens if valid_tokens > 0 else metrics_weight)

        metrics.update(
            {
                f"{key_prefix}hit_rate": 0.0,
                f"{key_prefix}prob_mean": 0.0,
                f"{key_prefix}neglogp": 0.0,
                f"{key_prefix}token_frac": 0.0,
            }
        )
        if expect_targets and on_missing is not None:
            on_missing("targets were all pad ids")
        return metrics_weight

    if expect_targets:
        metrics.update(
            {
                f"{key_prefix}hit_rate": 0.0,
                f"{key_prefix}prob_mean": 0.0,
                f"{key_prefix}neglogp": 0.0,
                f"{key_prefix}token_frac": 0.0,
            }
        )
        if on_missing is not None:
            if language_ids is None:
                reason = "no language_ids tensor provided"
            elif torch.is_tensor(language_ids) and (language_ids >= 0).any():
                reason = "language_ids could not be mapped to targets"
            else:
                reason = "language_ids contained only pad ids"
            on_missing(reason)
    return metrics_weight
