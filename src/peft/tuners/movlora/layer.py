from abc import ABC
from typing import Optional

import torch
import torch.nn as nn

from ..ia3 import IA3Layer
from ...metrics import record_movlora_metrics


class MovLoraLayer(IA3Layer, ABC):
    """MoV-style routed IA3-vector scaling layer.

    This keeps the layer wrapper and config plumbing from LoRA, but the adapted
    computation is IA3-style multiplicative scaling mixed across experts.
    """

    adapter_layer_names = ("lora_router", "lora_mov_scaling")
    other_param_names = (
        "num_experts",
        "router_top_k",
        "router_temperature",
        "router_jitter_noise",
        "router_ignore_padding_tokens",
        "adapter_is_feedforward",
    )

    def __init__(self, base_layer: nn.Module, is_feedforward: bool = False, **kwargs):
        IA3Layer.__init__(self, base_layer=base_layer, is_feedforward=is_feedforward, **kwargs)
        self.lora_router = nn.ModuleDict({})
        self.lora_mov_scaling = nn.ParameterDict({})
        self.num_experts = {}
        self.router_top_k = {}
        self.router_temperature = {}
        self.router_jitter_noise = {}
        self.router_ignore_padding_tokens = {}
        self.adapter_is_feedforward = {}

    def update_layer(
        self,
        adapter_name: str,
        init_ia3_weights: bool,
        num_experts: int,
        router_top_k: int,
        router_temperature: float,
        router_jitter_noise: float,
        router_bias: bool,
        router_init_std: float,
        router_ignore_padding_tokens: bool,
        is_feedforward: bool,
    ) -> None:
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

        self.num_experts[adapter_name] = num_experts
        self.router_top_k[adapter_name] = router_top_k
        self.router_temperature[adapter_name] = router_temperature
        self.router_jitter_noise[adapter_name] = router_jitter_noise
        self.router_ignore_padding_tokens[adapter_name] = router_ignore_padding_tokens
        self.adapter_is_feedforward[adapter_name] = is_feedforward

        mov_dim = self.in_features if is_feedforward else self.out_features
        if init_ia3_weights:
            scaling = torch.ones((num_experts, mov_dim), dtype=torch.float32)
        else:
            scaling = torch.randn((num_experts, mov_dim), dtype=torch.float32)
        self.lora_mov_scaling[adapter_name] = nn.Parameter(scaling)
        self.lora_router[adapter_name] = nn.Linear(self.in_features, num_experts, bias=router_bias)

        nn.init.normal_(self.lora_router[adapter_name].weight, std=router_init_std)
        if self.lora_router[adapter_name].bias is not None:
            nn.init.zeros_(self.lora_router[adapter_name].bias)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

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
            _, topk_indices = torch.topk(probs, k=top_k, dim=-1)
            topk_mask = torch.zeros_like(probs)
            topk_mask.scatter_(1, topk_indices, 1.0)
            # Match original MoV/MoLoRA top-k behavior: post-softmax masking.
            probs = probs * topk_mask

        if self.router_ignore_padding_tokens[adapter_name]:
            non_padding = (flat_inputs.abs().sum(dim=-1, keepdim=True) > 0).to(probs.dtype)
            probs = probs * non_padding

        return probs.reshape(*batch_shape, self.num_experts[adapter_name])

    def _record_routing_metrics(self, adapter_name: str, routing: torch.Tensor) -> None:
        with torch.no_grad():
            num_experts = int(self.num_experts[adapter_name])
            flat_routing = routing.detach().to(torch.float32).reshape(-1, num_experts)
            token_count = flat_routing.size(0)
            if token_count == 0:
                return

            row_sum = flat_routing.sum(dim=-1)
            active_mask = row_sum > 0
            active_tokens = int(active_mask.sum().item())
            zero_tokens = token_count - active_tokens

            experts_per_token = (flat_routing > 0).sum(dim=-1).to(torch.float32)
            expert_load = flat_routing.sum(dim=0)
            mean_load = expert_load.mean().item()
            if mean_load > 0:
                load_cv = float((expert_load.std(unbiased=False) / (mean_load + 1e-6)).item())
            else:
                load_cv = 0.0

            total_load = expert_load.sum().item()
            if total_load > 0:
                load_frac = expert_load / total_load
                load_max = float(load_frac.max().item())
                load_min = float(load_frac.min().item())
            else:
                load_max = 0.0
                load_min = 0.0

            if active_tokens > 0:
                normalized = flat_routing[active_mask] / (row_sum[active_mask].unsqueeze(-1) + 1e-8)
                entropy = float((-normalized * torch.log(normalized + 1e-8)).sum(dim=-1).mean().item())
                max_prob = float(normalized.max(dim=-1).values.mean().item())
            else:
                entropy = 0.0
                max_prob = 0.0

            record_movlora_metrics(
                {
                    "active_token_pct": active_tokens / float(token_count),
                    "zero_routed_pct": zero_tokens / float(token_count),
                    "experts_per_token": float(experts_per_token.mean().item()),
                    "expert_load_cv": load_cv,
                    "expert_load_max_frac": load_max,
                    "expert_load_min_frac": load_min,
                    "router_entropy": entropy,
                    "router_max_prob": max_prob,
                    "topk_density": float(experts_per_token.mean().item()) / max(float(num_experts), 1.0),
                },
                weight=float(token_count),
            )


class LinearMovLoraLayer(nn.Module, MovLoraLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        fan_in_fan_out: bool = False,
        is_feedforward: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_ia3_weights: bool = True,
        num_experts: int = 30,
        router_top_k: int = 0,
        router_temperature: float = 1.0,
        router_jitter_noise: float = 0.0,
        router_bias: bool = False,
        router_init_std: float = 2e-2,
        router_ignore_padding_tokens: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.fan_in_fan_out = fan_in_fan_out
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        MovLoraLayer.__init__(self, base_layer=base_layer, is_feedforward=is_feedforward, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            init_ia3_weights=init_ia3_weights,
            num_experts=num_experts,
            router_top_k=router_top_k,
            router_temperature=router_temperature,
            router_jitter_noise=router_jitter_noise,
            router_bias=router_bias,
            router_init_std=router_init_std,
            router_ignore_padding_tokens=router_ignore_padding_tokens,
            is_feedforward=is_feedforward,
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        del safe_merge
        del adapter_names
        raise NotImplementedError("MoV layers do not support merge/unmerge in this implementation.")

    def unmerge(self) -> None:
        raise NotImplementedError("MoV layers do not support merge/unmerge in this implementation.")

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            return self.base_layer(x, *args, **kwargs).to(previous_dtype)

        input_scaling = None
        output_scaling = None
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_mov_scaling:
                continue

            x_cast = x.to(self.lora_router[active_adapter].weight.dtype)
            routing = self._compute_routing_weights(active_adapter, x_cast)
            if self.training:
                self._record_routing_metrics(active_adapter, routing)

            scaling_bank = self.lora_mov_scaling[active_adapter]
            mixed_scaling = torch.einsum("...e,ed->...d", routing.to(scaling_bank.dtype), scaling_bank)

            if self.adapter_is_feedforward[active_adapter]:
                adapter_scale = mixed_scaling.to(x.dtype)
                input_scaling = adapter_scale if input_scaling is None else input_scaling * adapter_scale
            else:
                adapter_scale = mixed_scaling
                output_scaling = adapter_scale if output_scaling is None else output_scaling * adapter_scale

        scaled_input = x if input_scaling is None else x * input_scaling.to(x.dtype)
        result = self.base_layer(scaled_input, *args, **kwargs)
        if output_scaling is not None:
            result = result * output_scaling.to(result.dtype)

        return result.to(previous_dtype)
