import math
from abc import ABC
from typing import Optional

import torch
import torch.nn as nn

from ..lora import LoraLayer


class MovLoraLayer(LoraLayer, ABC):
    _PAPER_LORA_A_STD = 2e-2

    adapter_layer_names = LoraLayer.adapter_layer_names + ("lora_router",)
    other_param_names = LoraLayer.other_param_names + (
        "num_experts",
        "router_top_k",
        "router_temperature",
        "router_jitter_noise",
        "router_ignore_padding_tokens",
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__(base_layer, **kwargs)
        self.lora_router = nn.ModuleDict({})
        self.num_experts = {}
        self.router_top_k = {}
        self.router_temperature = {}
        self.router_jitter_noise = {}
        self.router_ignore_padding_tokens = {}

    def update_layer(
        self,
        adapter_name: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: bool,
        num_experts: int,
        router_top_k: int,
        router_temperature: float,
        router_jitter_noise: float,
        router_bias: bool,
        router_init_std: float,
        router_ignore_padding_tokens: bool,
        use_rslora: bool,
    ) -> None:
        if lora_rank <= 0:
            raise ValueError(f"The rank `r` should be a positive integer value but got {lora_rank}.")
        if num_experts <= 0:
            raise ValueError(f"`num_experts` must be positive, got {num_experts}.")
        if router_top_k < 0 or router_top_k > num_experts:
            raise ValueError(f"`router_top_k` must be in [0, {num_experts}], got {router_top_k}.")
        if router_temperature <= 0:
            raise ValueError(f"`router_temperature` must be positive, got {router_temperature}.")
        if router_jitter_noise < 0:
            raise ValueError(f"`router_jitter_noise` must be non-negative, got {router_jitter_noise}.")
        if router_init_std < 0:
            raise ValueError(f"`router_init_std` must be non-negative, got {router_init_std}.")

        self.r[adapter_name] = lora_rank
        self.lora_alpha[adapter_name] = lora_alpha
        self.num_experts[adapter_name] = num_experts
        self.router_top_k[adapter_name] = router_top_k
        self.router_temperature[adapter_name] = router_temperature
        self.router_jitter_noise[adapter_name] = router_jitter_noise
        self.router_ignore_padding_tokens[adapter_name] = router_ignore_padding_tokens

        if lora_dropout > 0.0:
            dropout_layers = nn.ModuleList([nn.Dropout(p=lora_dropout) for _ in range(num_experts)])
        else:
            dropout_layers = nn.ModuleList([nn.Identity() for _ in range(num_experts)])

        self.lora_dropout[adapter_name] = dropout_layers
        self.lora_A[adapter_name] = nn.ModuleList(
            [nn.Linear(self.in_features, lora_rank, bias=False) for _ in range(num_experts)]
        )
        self.lora_B[adapter_name] = nn.ModuleList(
            [nn.Linear(lora_rank, self.out_features, bias=False) for _ in range(num_experts)]
        )
        self.lora_router[adapter_name] = nn.Linear(self.in_features, num_experts, bias=router_bias)

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(lora_rank)
        else:
            self.scaling[adapter_name] = lora_alpha / lora_rank

        self.reset_lora_parameters(adapter_name, init_lora_weights)
        nn.init.normal_(self.lora_router[adapter_name].weight, std=router_init_std)
        if self.lora_router[adapter_name].bias is not None:
            nn.init.zeros_(self.lora_router[adapter_name].bias)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights) -> None:
        if init_lora_weights is False:
            return
        if adapter_name not in self.lora_A:
            return

        for expert_idx in range(self.num_experts[adapter_name]):
            if init_lora_weights is True:
                # Paper-faithful default used by the original MoLoRA implementation.
                nn.init.normal_(self.lora_A[adapter_name][expert_idx].weight, std=self._PAPER_LORA_A_STD)
            elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name][expert_idx].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unsupported MoV-LoRA initialization: {init_lora_weights}")
            nn.init.zeros_(self.lora_B[adapter_name][expert_idx].weight)

    def _compute_routing_weights(self, adapter_name: str, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        flat_inputs = x.reshape(-1, x.shape[-1])

        jitter_noise = self.router_jitter_noise[adapter_name]
        if self.training and jitter_noise > 0:
            flat_inputs = flat_inputs * torch.empty_like(flat_inputs).uniform_(1.0 - jitter_noise, 1.0 + jitter_noise)

        logits = self.lora_router[adapter_name](flat_inputs).float() / self.router_temperature[adapter_name]

        probs = torch.softmax(logits, dim=-1)

        top_k = self.router_top_k[adapter_name]
        if 0 < top_k < self.num_experts[adapter_name]:
            # Match original MoLoRA behavior: mask probabilities after softmax.
            _, topk_indices = torch.topk(probs, k=top_k, dim=-1)
            topk_mask = torch.zeros_like(probs)
            topk_mask.scatter_(1, topk_indices, 1.0)
            probs = probs * topk_mask

        if self.router_ignore_padding_tokens[adapter_name]:
            non_padding = (flat_inputs.abs().sum(dim=-1, keepdim=True) > 0).to(probs.dtype)
            probs = probs * non_padding

        return probs.reshape(*batch_shape, self.num_experts[adapter_name])


class LinearMovLoraLayer(nn.Module, MovLoraLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        lora_rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        num_experts: int = 8,
        router_top_k: int = 0,
        router_temperature: float = 1.0,
        router_jitter_noise: float = 0.0,
        router_bias: bool = False,
        router_init_std: float = 2e-2,
        router_ignore_padding_tokens: bool = False,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        MovLoraLayer.__init__(self, base_layer=base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            num_experts=num_experts,
            router_top_k=router_top_k,
            router_temperature=router_temperature,
            router_jitter_noise=router_jitter_noise,
            router_bias=router_bias,
            router_init_std=router_init_std,
            router_ignore_padding_tokens=router_ignore_padding_tokens,
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

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue

            x_cast = x.to(self.lora_A[active_adapter][0].weight.dtype)
            routing = self._compute_routing_weights(active_adapter, x_cast)
            scale = self.scaling[active_adapter]

            for expert_idx in range(self.num_experts[active_adapter]):
                delta = self.lora_B[active_adapter][expert_idx](
                    self.lora_A[active_adapter][expert_idx](self.lora_dropout[active_adapter][expert_idx](x_cast))
                )
                gate = routing[..., expert_idx].unsqueeze(-1).to(delta.dtype)
                result = result + (delta * gate * scale).to(result.dtype)

        return result.to(previous_dtype)
