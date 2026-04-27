import math
from abc import ABC
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..lora import LoraLayer


class VanillaMoELoraLayer(LoraLayer, ABC):
    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__(base_layer, **kwargs)
        self.num_experts = {}
        self.top_k = {}
        self.router_aux_loss_coef = {}
        self.lora_router = nn.ModuleDict({})
        self._latest_router_logits = {}

    def update_layer(
        self,
        adapter_name: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: bool,
        num_experts: int,
        top_k: int,
        router_init_range: float,
        router_aux_loss_coef: float,
        use_rslora: bool,
    ) -> None:
        if lora_rank <= 0:
            raise ValueError(f"The rank `r` should be positive, got {lora_rank}.")
        if num_experts <= 0:
            raise ValueError(f"`num_experts` must be positive, got {num_experts}.")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"`top_k` must be in [1, num_experts], got {top_k}.")

        self.r[adapter_name] = lora_rank
        self.lora_alpha[adapter_name] = lora_alpha
        self.num_experts[adapter_name] = num_experts
        self.top_k[adapter_name] = top_k
        self.router_aux_loss_coef[adapter_name] = router_aux_loss_coef
        self._latest_router_logits[adapter_name] = None

        dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_dropout[adapter_name] = dropout
        self.lora_A[adapter_name] = nn.ModuleList(
            [nn.Linear(self.in_features, lora_rank, bias=False) for _ in range(num_experts)]
        )
        self.lora_B[adapter_name] = nn.ModuleList(
            [nn.Linear(lora_rank, self.out_features, bias=False) for _ in range(num_experts)]
        )
        self.lora_router[adapter_name] = nn.Linear(self.in_features, num_experts, bias=False)
        nn.init.normal_(self.lora_router[adapter_name].weight, mean=0.0, std=router_init_range)

        self.scaling[adapter_name] = lora_alpha / math.sqrt(lora_rank) if use_rslora else lora_alpha / lora_rank
        self.reset_lora_parameters(adapter_name, init_lora_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights) -> None:
        if init_lora_weights is False or adapter_name not in self.lora_A:
            return
        for expert_idx in range(self.num_experts[adapter_name]):
            if init_lora_weights is True:
                nn.init.kaiming_uniform_(self.lora_A[adapter_name][expert_idx].weight, a=math.sqrt(5))
            elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name][expert_idx].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unsupported Vanilla MoE-LoRA initialization: {init_lora_weights}")
            nn.init.zeros_(self.lora_B[adapter_name][expert_idx].weight)

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str) -> None:
        super()._move_adapter_to_device_of_base_layer(adapter_name)
        weight = self.get_base_layer().weight
        self.lora_router[adapter_name] = self.lora_router[adapter_name].to(device=weight.device, dtype=weight.dtype)

    def pop_router_logits(self, adapter_name: str) -> Optional[torch.Tensor]:
        router_logits = self._latest_router_logits.get(adapter_name)
        self._latest_router_logits[adapter_name] = None
        return router_logits


class LinearVanillaMoELoraLayer(nn.Module, VanillaMoELoraLayer):
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
        router_init_range: float = 0.02,
        router_aux_loss_coef: float = 0.001,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        VanillaMoELoraLayer.__init__(self, base_layer=base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            num_experts=num_experts,
            top_k=top_k,
            router_init_range=router_init_range,
            router_aux_loss_coef=router_aux_loss_coef,
            use_rslora=use_rslora,
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        pass

    def unmerge(self) -> None:
        pass

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)
        if self.disable_adapters:
            return result.to(previous_dtype)

        batch_shape = result.shape[:-1]
        flat_result_shape = (result.reshape(-1, result.shape[-1]).shape[0], result.shape[-1])

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue

            router = self.lora_router[active_adapter]
            flat_x = x.reshape(-1, x.shape[-1])
            router_logits = router(flat_x.to(router.weight.dtype))
            router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            routing_weights, selected_experts = torch.topk(router_probs, self.top_k[active_adapter], dim=-1)
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            self._latest_router_logits[active_adapter] = router_logits

            delta_result = torch.zeros(flat_result_shape, dtype=result.dtype, device=result.device)
            x_cast = flat_x.to(self.lora_A[active_adapter][0].weight.dtype)
            for expert_idx in range(self.num_experts[active_adapter]):
                token_idx, kth_idx = torch.where(selected_experts == expert_idx)
                if token_idx.numel() == 0:
                    continue
                expert_delta = self.lora_B[active_adapter][expert_idx](
                    self.lora_A[active_adapter][expert_idx](self.lora_dropout[active_adapter](x_cast[token_idx]))
                )
                expert_delta = expert_delta * self.scaling[active_adapter]
                weights = routing_weights[token_idx, kth_idx].to(expert_delta.dtype).unsqueeze(-1)
                delta_result.index_add_(0, token_idx, (expert_delta * weights).to(delta_result.dtype))

            result = result + delta_result.reshape(*batch_shape, result.shape[-1])

        return result.to(previous_dtype)
