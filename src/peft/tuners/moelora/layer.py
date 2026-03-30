import math
import weakref
from abc import ABC
from typing import Optional

import torch
import torch.nn as nn

from ..lora import LoraLayer


class MoELoraLayer(LoraLayer, ABC):
    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__(base_layer, **kwargs)
        self.expert_num = {}
        self.expert_rank = {}
        self._moelora_parent_ref = None

    def set_moelora_parent(self, parent_model: nn.Module) -> None:
        self._moelora_parent_ref = weakref.ref(parent_model)

    def update_layer(
        self,
        adapter_name: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: bool,
        expert_num: int,
        use_rslora: bool,
    ) -> None:
        if lora_rank <= 0:
            raise ValueError(f"The rank `r` should be a positive integer value but got {lora_rank}.")
        if expert_num <= 0:
            raise ValueError(f"`expert_num` must be positive, got {expert_num}.")
        if lora_rank % expert_num != 0:
            raise ValueError(
                f"MoE-LoRA requires `r` divisible by `expert_num` for equal expert rank. Got r={lora_rank}, expert_num={expert_num}."
            )

        rank_per_expert = lora_rank // expert_num
        self.r[adapter_name] = lora_rank
        self.lora_alpha[adapter_name] = lora_alpha
        self.expert_num[adapter_name] = expert_num
        self.expert_rank[adapter_name] = rank_per_expert

        if lora_dropout > 0.0:
            dropout_layers = nn.ModuleList([nn.Dropout(p=lora_dropout) for _ in range(expert_num)])
        else:
            dropout_layers = nn.ModuleList([nn.Identity() for _ in range(expert_num)])

        self.lora_dropout[adapter_name] = dropout_layers
        self.lora_A[adapter_name] = nn.ModuleList(
            [nn.Linear(self.in_features, rank_per_expert, bias=False) for _ in range(expert_num)]
        )
        self.lora_B[adapter_name] = nn.ModuleList(
            [nn.Linear(rank_per_expert, self.out_features, bias=False) for _ in range(expert_num)]
        )

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(lora_rank)
        else:
            self.scaling[adapter_name] = lora_alpha / lora_rank

        self.reset_lora_parameters(adapter_name, init_lora_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights) -> None:
        if init_lora_weights is False:
            return
        if adapter_name not in self.lora_A:
            return

        for idx in range(self.expert_num[adapter_name]):
            if init_lora_weights is True:
                nn.init.kaiming_uniform_(self.lora_A[adapter_name][idx].weight, a=math.sqrt(5))
            elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name][idx].weight, std=1 / self.expert_rank[adapter_name])
            else:
                raise ValueError(f"Unsupported MoE-LoRA initialization: {init_lora_weights}")
            nn.init.zeros_(self.lora_B[adapter_name][idx].weight)

    def _get_routing_weights(self, adapter_name: str, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        parent_model = None if self._moelora_parent_ref is None else self._moelora_parent_ref()
        if parent_model is None:
            routing = torch.zeros(batch_size, self.expert_num[adapter_name], device=device, dtype=dtype)
            routing[:, 0] = 1.0
            return routing
        return parent_model.get_routing_weights(
            adapter_name=adapter_name, batch_size=batch_size, device=device, dtype=dtype
        )


class LinearMoELoraLayer(nn.Module, MoELoraLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        lora_rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        expert_num: int = 8,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        MoELoraLayer.__init__(self, base_layer=base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            expert_num=expert_num,
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

        batch_size = x.shape[0]
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue

            routing = self._get_routing_weights(
                adapter_name=active_adapter,
                batch_size=batch_size,
                device=x.device,
                dtype=result.dtype,
            )
            routing = routing.to(result.dtype)
            x_cast = x.to(self.lora_A[active_adapter][0].weight.dtype)
            scale = self.scaling[active_adapter]
            gate_eps = 1e-6
            active_experts = torch.nonzero((routing > gate_eps).any(dim=0), as_tuple=False).flatten()
            if active_experts.numel() == 0:
                continue
            for expert_idx in active_experts.tolist():
                gate = routing[:, expert_idx]
                active = gate > gate_eps
                if not torch.any(active):
                    continue

                if torch.all(active):
                    delta = self.lora_B[active_adapter][expert_idx](
                        self.lora_A[active_adapter][expert_idx](self.lora_dropout[active_adapter][expert_idx](x_cast))
                    )
                    gate_view = gate.view(batch_size, *([1] * (delta.dim() - 1))).to(delta.dtype)
                    result = result + (delta * gate_view * scale).to(result.dtype)
                    continue

                x_sub = x_cast[active]
                if x_sub.numel() == 0:
                    continue

                delta = self.lora_B[active_adapter][expert_idx](
                    self.lora_A[active_adapter][expert_idx](self.lora_dropout[active_adapter][expert_idx](x_sub))
                )
                gate_sub = gate[active].view(-1, *([1] * (delta.dim() - 1))).to(delta.dtype)
                result_sub = result[active] + (delta * gate_sub * scale).to(result.dtype)
                result[active] = result_sub

        return result.to(previous_dtype)
