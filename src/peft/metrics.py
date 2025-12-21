from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import torch


@dataclass
class _ScalarStat:
    total: float = 0.0
    weight: float = 0.0

    def update(self, value: float, weight: float) -> None:
        if not math.isfinite(value):
            return
        self.total += float(value) * weight
        self.weight += weight

    def mean(self) -> Optional[float]:
        if self.weight <= 0.0:
            return None
        return self.total / self.weight


class _ScalarAccumulator:
    def __init__(self) -> None:
        self._values: Dict[str, Dict[str, _ScalarStat]] = defaultdict(dict)

    def update(self, namespace: str, metrics: Dict[str, float], weight: float) -> None:
        if weight <= 0:
            weight = 1.0
        for key, value in metrics.items():
            if value is None:
                continue
            scoped = self._values[namespace].setdefault(key, _ScalarStat())
            scoped.update(float(value), weight)

    def pop(self) -> Dict[str, float]:
        aggregated: Dict[str, float] = {}
        for namespace, stats in self._values.items():
            prefix = f"{namespace}/" if namespace else ""
            for key, stat in stats.items():
                mean = stat.mean()
                if mean is None:
                    continue
                aggregated[f"{prefix}{key}"] = mean
        self._values.clear()
        return aggregated


@dataclass
class _MoelprRoutingAccumulator:
    expert_language: Dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    expert_totals: Counter = field(default_factory=Counter)
    original_counts: Counter = field(default_factory=Counter)
    new_counts: Counter = field(default_factory=Counter)
    original_total: float = 0.0
    new_total: float = 0.0
    num_experts: int = 0

    def update(
        self,
        language_ids: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        num_experts: int,
        lang_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if language_ids.numel() == 0 or selected_experts.numel() == 0:
            return

        self.num_experts = max(self.num_experts, int(num_experts))
        top_k = selected_experts.size(1)

        lang_cpu = language_ids.to(torch.long).cpu()
        experts_cpu = selected_experts.to(torch.long).cpu()
        weights_cpu = routing_weights.to(torch.float32).cpu()

        flat_lang = lang_cpu.repeat_interleave(top_k)
        flat_experts = experts_cpu.reshape(-1)
        flat_weights = weights_cpu.reshape(-1)

        positive = flat_weights > 0
        if torch.any(positive):
            flat_lang = flat_lang[positive]
            flat_experts = flat_experts[positive]
            flat_weights = flat_weights[positive]

        if flat_lang.numel() == 0:
            return

        pairs = torch.stack((flat_experts, flat_lang), dim=1)
        unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
        sums = torch.zeros(unique_pairs.size(0), dtype=torch.float32)
        sums.scatter_add_(0, inverse, flat_weights)

        for idx, (expert_idx, lang_id) in enumerate(unique_pairs.tolist()):
            count = float(sums[idx].item())
            if count <= 0:
                continue
            self.expert_language[int(expert_idx)][int(lang_id)] += count
            self.expert_totals[int(expert_idx)] += count

        if lang_mask is not None and lang_mask.numel() > 0:
            mask_cpu = lang_mask.to(torch.bool).cpu()
            primary = experts_cpu[:, 0]
            if mask_cpu.sum().item() > 0:
                orig_counts = torch.bincount(primary[mask_cpu], minlength=num_experts).tolist()
                for idx, count in enumerate(orig_counts):
                    if count == 0:
                        continue
                    self.original_counts[idx] += float(count)
                self.original_total += float(sum(orig_counts))
            new_mask = ~mask_cpu
            if new_mask.sum().item() > 0:
                new_counts = torch.bincount(primary[new_mask], minlength=num_experts).tolist()
                for idx, count in enumerate(new_counts):
                    if count == 0:
                        continue
                    self.new_counts[idx] += float(count)
                self.new_total += float(sum(new_counts))

    def pop(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.num_experts > 0:
            for expert_idx in range(self.num_experts):
                total = float(self.expert_totals.get(expert_idx, 0.0))
                lang_usage = self.expert_language.get(expert_idx)
                if not lang_usage or total <= 0:
                    continue
                top_lang_id, top_lang_count = lang_usage.most_common(1)[0]
                probs = [count / total for count in lang_usage.values() if count > 0]
                entropy = -sum(p * math.log(p + 1e-8) for p in probs)
                metrics[f"moelpr/expert_{expert_idx}_top_lang_id"] = float(top_lang_id)
                metrics[f"moelpr/expert_{expert_idx}_top_lang_frac"] = top_lang_count / total
                metrics[f"moelpr/expert_{expert_idx}_lang_entropy"] = entropy
                metrics[f"moelpr/expert_{expert_idx}_num_langs"] = float(len(lang_usage))

        if self.original_total > 0:
            base_share = float(self.original_counts.get(0, 0.0)) / self.original_total
            top = max(self.original_counts.items(), key=lambda kv: kv[1], default=(None, 0.0))
            if top[0] is not None and top[1] > 0:
                metrics["moelpr/original_primary_expert"] = float(top[0])
                metrics["moelpr/original_primary_frac"] = float(top[1]) / self.original_total
            metrics["moelpr/original_to_base_pct"] = base_share

        if self.new_total > 0:
            top = max(self.new_counts.items(), key=lambda kv: kv[1], default=(None, 0.0))
            if top[0] is not None and top[1] > 0:
                metrics["moelpr/new_primary_expert"] = float(top[0])
                metrics["moelpr/new_primary_frac"] = float(top[1]) / self.new_total

        self.reset()
        return metrics

    def reset(self) -> None:
        self.expert_language = defaultdict(Counter)
        self.expert_totals = Counter()
        self.original_counts = Counter()
        self.new_counts = Counter()
        self.original_total = 0.0
        self.new_total = 0.0
        self.num_experts = 0


_SCALAR_STORE = _ScalarAccumulator()
_MOELPR_ROUTING = _MoelprRoutingAccumulator()


def record_mola_metrics(metrics: Dict[str, float], weight: float) -> None:
    _SCALAR_STORE.update("mola", metrics, weight)


def record_adamole_metrics(metrics: Dict[str, float], weight: float) -> None:
    _SCALAR_STORE.update("adamole", metrics, weight)


def record_moelpr_scalars(metrics: Dict[str, float], weight: float) -> None:
    _SCALAR_STORE.update("moelpr", metrics, weight)


def record_cola_metrics(metrics: Dict[str, float], weight: float) -> None:
    _SCALAR_STORE.update("cola", metrics, weight)


def record_hydralora_metrics(metrics: Dict[str, float], weight: float) -> None:
    _SCALAR_STORE.update("hydralora", metrics, weight)


def record_moelpr_language_routing(
    language_ids: Optional[torch.Tensor],
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
    *,
    lang_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> None:
    if language_ids is None:
        return

    lang = language_ids
    if lang.dim() != 1:
        lang = lang.reshape(-1)

    experts = selected_experts
    weights = routing_weights

    mask = None
    if attention_mask is not None:
        attn = attention_mask
        if attn.dim() != 1:
            attn = attn.reshape(-1)
        mask = attn > 0

    if mask is not None:
        if mask.shape[0] != lang.shape[0]:
            mask = mask[: lang.shape[0]]
        lang = lang[mask]
        experts = experts[mask]
        weights = weights[mask]
        if lang_mask is not None:
            lang_mask = lang_mask.reshape(-1)[mask]

    valid = lang >= 0
    if not torch.any(valid):
        return

    lang = lang[valid]
    experts = experts[valid]
    weights = weights[valid]
    mask_subset = lang_mask[valid] if lang_mask is not None else None

    if lang.numel() == 0:
        return

    _MOELPR_ROUTING.update(lang, experts, weights, num_experts, mask_subset)


def pop_tracked_metrics() -> Dict[str, float]:
    metrics = _SCALAR_STORE.pop()
    routing_metrics = _MOELPR_ROUTING.pop()
    metrics.update(routing_metrics)
    return metrics
