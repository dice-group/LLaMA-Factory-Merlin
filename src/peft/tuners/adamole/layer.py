import math
from abc import ABC
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..lora import LoraLayer
from ...metrics import record_adamole_metrics


def _debug_print(msg: str) -> None:
    print(msg)


class LoraExpert(nn.Module):
    """
    Simple LoRA expert that reuses the projected A/B matrices.
    """

    def __init__(self, lora_A: nn.Module, lora_B: nn.Module, lora_dropout: nn.Module, scaling: float) -> None:
        super().__init__()
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.lora_dropout = lora_dropout
        self.scaling = scaling

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dropped = self.lora_dropout(inputs)
        return self.lora_B(self.lora_A(dropped)) * self.scaling


class AdaMoeLayer(nn.Module):
    """
    Adaptive Mixture of Experts (MoE) Layer
    """

    def __init__(
        self,
        experts: nn.ModuleList,
        gate: nn.Module,
        threshold_fn: nn.Module,
        max_threshold: float,
        debug_mode: bool = False,
    ):
        super().__init__()
        self.experts = experts
        self.gate = gate
        self.threshold_fn = threshold_fn
        self.max_threshold = max_threshold
        self.layer_loss = None
        self.debug_mode = debug_mode

    def get_layer_loss(self, gate_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """
        Get the load balancing loss by following the Switch Transformer
        """
        num_inputs = gate_logits.shape[0]
        num_experts = len(self.experts)
        expert_counts = torch.sum(selected_experts, dim=0)
        expert_fractions = expert_counts / num_inputs
        expert_probs = torch.sum(gate_logits, dim=0) / num_inputs
        layer_loss = num_experts * torch.sum(expert_fractions * expert_probs)
        return layer_loss

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation
        """
        flattened_inputs = inputs.view((-1, inputs.shape[-1]))
        gate_logits = F.softmax(self.gate(flattened_inputs), dim=-1)
        thresholds = torch.sigmoid(self.threshold_fn(flattened_inputs)) * self.max_threshold
        adapted_gate_logits = gate_logits - thresholds
        selected_experts = torch.ge(adapted_gate_logits, 0).to(torch.float)
        weights = adapted_gate_logits * selected_experts
        weight_sums = torch.sum(weights, dim=-1, keepdim=True, dtype=inputs.dtype)
        weight_sums = torch.where(weight_sums == 0, torch.ones_like(weight_sums), weight_sums)
        weights = weights / weight_sums
        results = torch.zeros_like(self.experts[0](flattened_inputs))

        for i, expert in enumerate(self.experts):
            batch_idx = torch.where(selected_experts[:, i])[0]
            if len(batch_idx) > 0:
                results[batch_idx] += weights[batch_idx, i, None] * expert(flattened_inputs[batch_idx])

        results = results.view((*inputs.shape[:-1], results.shape[-1]))
        if inputs.requires_grad:
            self.layer_loss = self.get_layer_loss(gate_logits=adapted_gate_logits, selected_experts=selected_experts)
        if self.debug_mode:
            with torch.no_grad():
                selected_counts = torch.sum(selected_experts, dim=0)
                token_selection = torch.sum(selected_experts, dim=1)
                _debug_print(
                    "[ADAMOLE DEBUG] gate_mean={:.4f} gate_std={:.4f} "
                    "thresh_mean={:.4f} thresh_std={:.4f} max_thresh={:.4f}".format(
                        gate_logits.mean().item(),
                        gate_logits.std().item(),
                        thresholds.mean().item(),
                        thresholds.std().item(),
                        float(self.max_threshold),
                    )
                )
                _debug_print(
                    "[ADAMOLE DEBUG] selected_expert_distribution={}".format(selected_counts.cpu().tolist())
                )
                _debug_print(
                    "[ADAMOLE DEBUG] experts_per_token min={} max={} mean={:.2f} zero_routed={}".format(
                        int(token_selection.min().item()),
                        int(token_selection.max().item()),
                        float(token_selection.float().mean().item()),
                        int((token_selection == 0).sum().item()),
                    )
                )
                if self.layer_loss is not None:
                    _debug_print("[ADAMOLE DEBUG] aux_loss={:.6f}".format(float(self.layer_loss.detach().item())))
        with torch.no_grad():
            token_count = selected_experts.size(0)
            if token_count > 0:
                experts_per_token = selected_experts.sum(dim=-1)
                active_tokens = (experts_per_token > 0).sum().item()
                zero_tokens = token_count - active_tokens
                expert_counts = selected_experts.sum(dim=0)
                load_cv = 0.0
                if expert_counts.numel() > 0:
                    mean_load = expert_counts.float().mean()
                    if mean_load > 0:
                        load_cv = float(
                            (expert_counts.float().std(unbiased=False) / (mean_load + 1e-6)).item()
                        )
                margin_vals = adapted_gate_logits[selected_experts.bool()]
                avg_margin = float(margin_vals.mean().item()) if margin_vals.numel() > 0 else 0.0
                record_adamole_metrics(
                    {
                        "active_token_pct": active_tokens / float(token_count),
                        "zero_routed_pct": zero_tokens / float(token_count),
                        "experts_per_token": float(experts_per_token.float().mean().item()),
                        "expert_load_cv": load_cv,
                        "threshold_mean": float(thresholds.mean().item()),
                        "threshold_std": float(thresholds.std(unbiased=False).item()),
                        "margin_mean": avg_margin,
                    },
                    weight=float(token_count),
                )
        return results


class AdaMoleLayer(LoraLayer, ABC):
    """
    AdaMoLE Layer
    """

    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__(base_layer, **kwargs)
        self.lora_gating = nn.ModuleDict({})
        self.lora_threshold = nn.ModuleDict({})
        self.moe_layer = nn.ModuleDict({})

    def update_layer(
        self,
        adapter_name: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: bool,
        num_experts: int,
        max_threshold: float,
        debug_mode: bool = False,
    ) -> None:
        """
        Update the layer
        """
        if lora_rank <= 0:
            raise ValueError(f"The rank `r` should be a positive integer value but the value passed is {lora_rank}.")

        if max_threshold is None:
            max_threshold = 1 / num_experts

        self.r[adapter_name] = lora_rank
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.ModuleList([nn.Dropout(p=lora_dropout) for _ in range(num_experts)])
        else:
            lora_dropout_layer = nn.ModuleList([nn.Identity() for _ in range(num_experts)])

        self.lora_dropout[adapter_name] = lora_dropout_layer
        self.lora_A[adapter_name] = nn.ModuleList(
            [nn.Linear(self.in_features, lora_rank, bias=False) for _ in range(num_experts)])
        self.lora_B[adapter_name] = nn.ModuleList(
            [nn.Linear(lora_rank, self.out_features, bias=False) for _ in range(num_experts)])
        self.scaling[adapter_name] = lora_alpha / lora_rank
        self.lora_gating[adapter_name] = nn.Linear(self.in_features, num_experts, bias=False)
        self.lora_threshold[adapter_name] = nn.Linear(self.in_features, 1)
        if debug_mode:
            _debug_print(
                "[ADAMOLE DEBUG] Adapter {} -> base={} in={} out={} rank={} alpha={} num_experts={} "
                "max_threshold={:.4f} dropout={:.2f}".format(
                    adapter_name,
                    self.base_layer.__class__.__name__,
                    self.in_features,
                    self.out_features,
                    lora_rank,
                    lora_alpha,
                    num_experts,
                    max_threshold,
                    lora_dropout,
                )
            )

        experts = nn.ModuleList([LoraExpert(
            self.lora_A[adapter_name][i],
            self.lora_B[adapter_name][i],
            self.lora_dropout[adapter_name][i],
            self.scaling[adapter_name],
        ) for i in range(num_experts)])
        self.moe_layer[adapter_name] = AdaMoeLayer(
            experts=experts,
            gate=self.lora_gating[adapter_name],
            threshold_fn=self.lora_threshold[adapter_name],
            max_threshold=max_threshold,
            debug_mode=debug_mode,
        )

        self.reset_parameters(adapter_name, init_lora_weights)
        self.set_adapter(self.active_adapters)

    def reset_parameters(self, adapter_name: str, init_lora_weights: bool) -> None:
        """
        Reset the parameters
        """
        if init_lora_weights is False:
            return
        elif adapter_name in self.lora_A.keys():
            for i in range(len(self.lora_A[adapter_name])):
                nn.init.kaiming_uniform_(self.lora_A[adapter_name][i].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[adapter_name][i].weight)


class LinearAdaMoleLayer(nn.Module, AdaMoleLayer):
    """
    AdaMoLE Implementation in a Linear Layer
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        lora_rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        num_experts: int = 4,
        max_threshold: float = None,
        debug_mode: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        AdaMoleLayer.__init__(self, base_layer=base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            lora_rank,
            lora_alpha,
            lora_dropout,
            init_lora_weights,
            num_experts,
            max_threshold,
            debug_mode=debug_mode,
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights
        """
        pass

    def unmerge(self) -> None:
        """
        Unmerge all merged adapter layers from the base weights
        """
        pass

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward propagation
        """
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            moe_layer = self.moe_layer[active_adapter]
            x = x.to(moe_layer.experts[0].lora_A.weight.dtype)
            result += moe_layer(x)

        result = result.to(previous_dtype)
        return result
