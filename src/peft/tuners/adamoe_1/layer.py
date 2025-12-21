import math
from abc import ABC
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..lora import LoraLayer
from ...metrics import record_mola_metrics


def _debug_print(msg: str) -> None:
    print(msg)


class NullExpert(nn.Module):
    """
    Null Expert
    """
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(inputs)


class TopKMoeLayer(nn.Module):
    """
    Mixture of Experts (MoE) Layer with the Top-k

    Adapted from https://github.com/mistralai/mistral-src
    """

    def __init__(self, experts: nn.ModuleList, gate: nn.Module, top_k: int, use_null_expert: bool, router_aux_loss_coef: float, null_expert_penalty: float, aux_loss_annealing: bool, debug_mode: bool):
        super().__init__()
        self.experts = experts
        self.gate = gate
        self.top_k = top_k
        self.use_null_expert = use_null_expert
        self.router_aux_loss_coef = router_aux_loss_coef
        self.null_expert_penalty = null_expert_penalty
        self.aux_loss_annealing = aux_loss_annealing
        self.debug_mode = debug_mode
        self.layer_loss = torch.tensor(0.0, device=next(gate.parameters()).device)

    def get_layer_loss(self, gate_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """
        Get the load balancing loss by following the Switch Transformer
        """
        num_inputs = gate_logits.shape[0]
        num_experts = len(self.experts)
        expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
        expert_fractions = expert_counts / num_inputs
        expert_probs = torch.sum(gate_logits, dim=0) / num_inputs
        if self.debug_mode:
            _debug_print(f"[MOLA DEBUG] expert_fractions: {expert_fractions}")
            _debug_print(f"[MOLA DEBUG] expert_probs: {expert_probs}")

        if self.use_null_expert:
            # AdaMoE loss
            # Last expert is the null expert
            true_expert_fractions = expert_fractions[:-1]
            true_expert_probs = expert_probs[:-1]
            null_expert_fraction = expert_fractions[-1]
            if self.debug_mode:
                _debug_print(f"[MOLA DEBUG] null_expert_fraction: {null_expert_fraction}")

            loss = self.router_aux_loss_coef * (torch.sum(true_expert_fractions * true_expert_probs) - self.null_expert_penalty * null_expert_fraction)
        else:
            # Standard load balancing loss
            loss = self.router_aux_loss_coef * num_experts * torch.sum(expert_fractions * expert_probs)

        if self.debug_mode:
            _debug_print(f"[MOLA DEBUG] aux_loss={loss}")
        return loss

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation
        """
        flattened_inputs = inputs.view((-1, inputs.shape[-1]))
        gate_logits = F.softmax(self.gate(flattened_inputs), dim=-1)
        if self.debug_mode:
            _debug_print(f"[MOLA DEBUG] self.gate.weight.requires_grad: {self.gate.weight.requires_grad}")
            _debug_print(f"[MOLA DEBUG] gate_logits.requires_grad: {gate_logits.requires_grad}")
        weights, selected_experts = torch.topk(input=gate_logits, k=self.top_k, dim=-1)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True, dtype=inputs.dtype)
        results = torch.zeros_like(self.experts[0](flattened_inputs))

        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            if self.use_null_expert and i == len(self.experts) - 1:
                # This is the null expert, do nothing
                continue
            results[batch_idx] += \
                weights[batch_idx, nth_expert, None] * expert(flattened_inputs[batch_idx])

        results = results.view((*inputs.shape[:-1], results.shape[-1]))
        self.layer_loss = self.get_layer_loss(gate_logits=gate_logits, selected_experts=selected_experts)
        if self.debug_mode:
            with torch.no_grad():
                expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=len(self.experts))
                null_count = expert_counts[-1].item() if self.use_null_expert else 0
                _debug_print(
                    "[MOLA DEBUG] tokens={} expert_counts={} null_expert_tokens={}".format(
                        flattened_inputs.shape[0],
                        expert_counts.cpu().tolist(),
                        null_count,
                    )
                )
                _debug_print("[MOLA DEBUG] aux_loss={:.6f}".format(float(self.layer_loss.detach().item())))
        with torch.no_grad():
            token_count = flattened_inputs.shape[0]
            assign_count = selected_experts.numel()
            if token_count > 0 and assign_count > 0:
                entropy = (-gate_logits.float() * torch.log(gate_logits.float() + 1e-8)).sum(dim=-1).mean()
                expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=len(self.experts))
                if self.use_null_expert and expert_counts.numel() > 0:
                    true_counts = expert_counts[:-1]
                    null_paths = float(expert_counts[-1].item())
                    tokens_with_null = float((selected_experts == len(self.experts) - 1).any(dim=-1).sum().item())
                else:
                    true_counts = expert_counts
                    null_paths = 0.0
                    tokens_with_null = 0.0
                load_cv = 0.0
                active_frac = 0.0
                if true_counts.numel() > 0:
                    mean_load = true_counts.float().mean()
                    if mean_load > 0:
                        load_cv = float((true_counts.float().std(unbiased=False) / (mean_load + 1e-6)).item())
                    active_frac = float((true_counts > 0).float().mean().item())
                record_mola_metrics(
                    {
                        "null_token_pct": tokens_with_null / float(token_count),
                        "null_path_pct": null_paths / float(assign_count),
                        "expert_load_cv": load_cv,
                        "router_entropy": float(entropy.item()),
                        "active_expert_frac": active_frac,
                    },
                    weight=float(token_count),
                )
        return results


class LoraExpert(nn.Module):
    """
    LoRA Expert
    """

    def __init__(self, lora_A: nn.Module, lora_B: nn.Module, lora_dropout: nn.Module, scaling: float):
        super().__init__()
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.lora_dropout = lora_dropout
        self.scaling = scaling

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation
        """
        outputs = self.lora_B(self.lora_A(self.lora_dropout(inputs))) * self.scaling
        return outputs


class MolaLayer(LoraLayer, ABC):
    """
    MoLA Layer
    """

    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__(base_layer, **kwargs)
        self.lora_gating = nn.ModuleDict({})
        self.moe_layer = nn.ModuleDict({})

    def update_layer(
        self, adapter_name: str, lora_rank: int, lora_alpha: int, lora_dropout: float, init_lora_weights: bool,
        num_experts: int, top_k: int, use_null_expert: bool, router_aux_loss_coef: float, null_expert_penalty: float, aux_loss_annealing: bool, mola_debug_mode: bool
    ) -> None:
        """
        Update the layer
        """
        if lora_rank <= 0:
            raise ValueError(f"The rank `r` should be a positive integer value but the value passed is {lora_rank}.")

        self.r[adapter_name] = lora_rank
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.ModuleList(nn.Dropout(p=lora_dropout) for _ in range(num_experts))
        else:
            lora_dropout_layer = nn.ModuleList(nn.Identity() for _ in range(num_experts))

        self.lora_dropout[adapter_name] = lora_dropout_layer.to(self.base_layer.weight.device)
        self.lora_A[adapter_name] = nn.ModuleList(
            nn.Linear(self.in_features, lora_rank, bias=False) for _ in range(num_experts)
        ).to(self.base_layer.weight.device)
        self.lora_B[adapter_name] = nn.ModuleList(
            nn.Linear(lora_rank, self.out_features, bias=False) for _ in range(num_experts)
        ).to(self.base_layer.weight.device)
        self.scaling[adapter_name] = lora_alpha / lora_rank
        
        gating_num_experts = num_experts
        if use_null_expert:
            gating_num_experts += 1
            
        self.lora_gating[adapter_name] = nn.Linear(self.in_features, gating_num_experts, bias=False).to(self.base_layer.weight.device)

        if mola_debug_mode:
            _debug_print(
                "[MOLA DEBUG] Adapter {} -> base={} in={} out={} rank={} alpha={} num_experts={} top_k={} "
                "use_null={} dropout={:.2f}".format(
                    adapter_name,
                    self.base_layer.__class__.__name__,
                    self.in_features,
                    self.out_features,
                    lora_rank,
                    lora_alpha,
                    num_experts,
                    top_k,
                    use_null_expert,
                    lora_dropout,
                )
            )

        experts = nn.ModuleList(LoraExpert(
            self.lora_A[adapter_name][i],
            self.lora_B[adapter_name][i],
            self.lora_dropout[adapter_name][i],
            self.scaling[adapter_name],
        ) for i in range(num_experts))
        
        if use_null_expert:
            experts.append(NullExpert().to(self.base_layer.weight.device))

        self.moe_layer[adapter_name] = TopKMoeLayer(
            experts=experts, gate=self.lora_gating[adapter_name], top_k=top_k, use_null_expert=use_null_expert,
            router_aux_loss_coef=router_aux_loss_coef, null_expert_penalty=null_expert_penalty, aux_loss_annealing=aux_loss_annealing, debug_mode=mola_debug_mode
        ).to(self.base_layer.weight.device)

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


class LinearMolaLayer(nn.Module, MolaLayer):
    """
    MoLA Implementation in a Linear Layer
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
        top_k: int = 2,
        use_null_expert: bool = False,
        router_aux_loss_coef: float = 0.01,
        null_expert_penalty: float = 0.1,
        aux_loss_annealing: bool = False,
        mola_debug_mode: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        MolaLayer.__init__(self, base_layer=base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, lora_rank, lora_alpha, lora_dropout, init_lora_weights, num_experts, top_k,
            use_null_expert, router_aux_loss_coef, null_expert_penalty, aux_loss_annealing, mola_debug_mode)

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
            x_for_moe = x.to(moe_layer.experts[0].lora_A.weight.device, dtype=moe_layer.experts[0].lora_A.weight.dtype)
            result = result + moe_layer(x_for_moe)

        result = result.to(x.device, dtype=previous_dtype)
        return result

# FOR DEBUG, TODO: remove if not necessary anymore
def check_tensor(name: str, t: torch.Tensor):
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"[NaN DETECTED] {name} "
                           f"dtype={t.dtype} "
                           f"max={t.max().item() if t.numel() > 0 and not torch.isnan(t).any() else 'nan'} "
                           f"min={t.min().item() if t.numel() > 0 and not torch.isnan(t).any() else 'nan'}")
