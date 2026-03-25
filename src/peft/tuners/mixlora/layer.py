from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers.activations import ACT2FN


class MixLoraExpertLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_rslora: bool,
        init_lora_weights: bool | str,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_rslora = use_rslora
        self.init_lora_weights = init_lora_weights

        self.r: dict[str, int] = {}
        self.lora_alpha: dict[str, int] = {}
        self.scaling: dict[str, float] = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})

    def update_layer(self, adapter_name: str, r: int, lora_alpha: int, lora_dropout: float) -> None:
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / math.sqrt(r) if self.use_rslora else lora_alpha / r
        self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        self.reset_lora_parameters(adapter_name, self.init_lora_weights)

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights: bool | str) -> None:
        if init_lora_weights is False:
            return

        if init_lora_weights is True:
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "gaussian":
            nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
        else:
            raise ValueError(f"Unknown initialization {init_lora_weights=}")

        nn.init.zeros_(self.lora_B[adapter_name].weight)

    def forward_delta(self, x: torch.Tensor, adapter_name: str) -> Optional[torch.Tensor]:
        """ computes expert specific lora correction / update """
        if adapter_name not in self.lora_A:
            return None

        lora_A = self.lora_A[adapter_name]
        lora_B = self.lora_B[adapter_name]
        hidden = self.lora_dropout[adapter_name](x)
        hidden = hidden.to(lora_A.weight.dtype)
        delta = lora_B(lora_A(hidden)) * self.scaling[adapter_name]
        return delta.to(x.dtype)


class MixLoraExpertGroup(nn.Module):
    """ Holds all LoRA experts for the target MLP projections in a MixLoRA layer """
    def __init__(
        self,
        projection_shapes: dict[str, tuple[int, int]],
        *,
        num_experts: int,
        use_rslora: bool,
        init_lora_weights: bool | str,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleDict({})
        for proj_name, (in_features, out_features) in projection_shapes.items():
            self.projections[proj_name] = nn.ModuleList(
                [
                    MixLoraExpertLinear(
                        in_features,
                        out_features,
                        use_rslora=use_rslora,
                        init_lora_weights=init_lora_weights,
                    )
                    for _ in range(num_experts)
                ]
            )

    def update_layer(self, adapter_name: str, r: int, lora_alpha: int, lora_dropout: float) -> None:
        for experts in self.projections.values():
            for expert in experts:
                expert.update_layer(adapter_name, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward_delta(self, projection_name: str, expert_idx: int, x: torch.Tensor, adapter_name: str) -> Optional[torch.Tensor]:
        if projection_name not in self.projections:
            return None
        experts = self.projections[projection_name]
        return experts[expert_idx].forward_delta(x, adapter_name)


class MixLoraMoeLayer(nn.Module, BaseTunerLayer):
    adapter_layer_names: tuple[str, ...] = ("lora_router",)
    other_param_names: tuple[str, ...] = ()

    def __init__(self, base_layer: nn.Module, adapter_name: str, mixlora_config) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.projection_shapes = self._infer_projection_shapes(base_layer, mixlora_config.moe_target_modules)
        self.input_features = self._infer_input_features(base_layer)
        self.output_features = self._infer_output_features(base_layer)

        self.lora_router = nn.ModuleDict({})
        self.experts = MixLoraExpertGroup(
            self.projection_shapes,
            num_experts=mixlora_config.num_experts,
            use_rslora=mixlora_config.use_rslora,
            init_lora_weights=mixlora_config.init_lora_weights,
        )

        self.num_experts: dict[str, int] = {}
        self.top_k: dict[str, int] = {}
        self.jitter_noise: dict[str, float] = {}
        self.router_loss: dict[str, bool] = {}
        self.router_aux_loss_coef: dict[str, float] = {}
        self._act_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        self._act_fn_names: dict[str, str] = {}
        self._latest_router_logits: dict[str, Optional[torch.Tensor]] = {}
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, mixlora_config)

    @staticmethod
    def _infer_input_features(base_layer: nn.Module) -> int:
        if hasattr(base_layer, "gate_proj"):
            return base_layer.gate_proj.in_features
        raise ValueError(f"Unsupported MixLoRA base layer type: {type(base_layer).__name__}")

    @staticmethod
    def _infer_output_features(base_layer: nn.Module) -> int:
        if hasattr(base_layer, "down_proj"):
            return base_layer.down_proj.out_features
        raise ValueError(f"Unsupported MixLoRA base layer type: {type(base_layer).__name__}")

    @staticmethod
    def _infer_projection_shapes(base_layer: nn.Module, target_modules) -> dict[str, tuple[int, int]]:
        if not all(hasattr(base_layer, name) for name in ("gate_proj", "up_proj", "down_proj")):
            raise ValueError(
                f"MixLoRA currently supports Llama-style MLP blocks with gate_proj/up_proj/down_proj, got {type(base_layer).__name__}."
            )

        available = {
            "gate_proj": (base_layer.gate_proj.in_features, base_layer.gate_proj.out_features),
            "up_proj": (base_layer.up_proj.in_features, base_layer.up_proj.out_features),
            "down_proj": (base_layer.down_proj.in_features, base_layer.down_proj.out_features),
        }
        selected = set(target_modules or available.keys())
        return {name: shape for name, shape in available.items() if name in selected}

    @staticmethod
    def _resolve_act_fn(base_layer: nn.Module, act_fn: Optional[str]) -> tuple[Callable[[torch.Tensor], torch.Tensor], str]:
        if act_fn is None:
            if not hasattr(base_layer, "act_fn"):
                raise ValueError(f"MixLoRA base layer {type(base_layer).__name__} has no `act_fn` to reuse.")
            base_act_fn = base_layer.act_fn
            return base_act_fn, type(base_act_fn).__name__

        if act_fn not in ACT2FN:
            raise ValueError(f"Unknown MixLoRA activation function: {act_fn}")

        return ACT2FN[act_fn], act_fn

    def _get_reference_weight(self) -> torch.Tensor:
        return self.base_layer.gate_proj.weight

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
        reference = self._get_reference_weight()
        if device is None:
            device = reference.device

        if adapter_name in self.lora_router:
            self.lora_router[adapter_name] = self.lora_router[adapter_name].to(device=device, dtype=reference.dtype)
        self.experts = self.experts.to(device=device, dtype=reference.dtype)

    def update_layer(self, adapter_name: str, mixlora_config) -> None:
        expert_rank = mixlora_config.expert_lora_r or mixlora_config.r
        expert_alpha = mixlora_config.expert_lora_alpha or mixlora_config.lora_alpha
        expert_dropout = (
            mixlora_config.expert_lora_dropout
            if mixlora_config.expert_lora_dropout is not None
            else mixlora_config.lora_dropout
        )

        self.num_experts[adapter_name] = mixlora_config.num_experts
        self.top_k[adapter_name] = mixlora_config.top_k
        self.jitter_noise[adapter_name] = mixlora_config.jitter_noise
        self.router_loss[adapter_name] = mixlora_config.router_loss
        self.router_aux_loss_coef[adapter_name] = mixlora_config.router_aux_loss_coef
        self._act_fns[adapter_name], self._act_fn_names[adapter_name] = self._resolve_act_fn(
            self.base_layer, mixlora_config.act_fn
        )

        self.lora_router[adapter_name] = nn.Linear(self.input_features, mixlora_config.num_experts, bias=False)
        nn.init.normal_(self.lora_router[adapter_name].weight, mean=0.0, std=mixlora_config.router_init_range)

        self.experts.update_layer(
            adapter_name,
            r=expert_rank,
            lora_alpha=expert_alpha,
            lora_dropout=expert_dropout,
        )
        self._latest_router_logits[adapter_name] = None
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def pop_router_logits(self, adapter_name: str) -> Optional[torch.Tensor]:
        router_logits = self._latest_router_logits.get(adapter_name)
        self._latest_router_logits[adapter_name] = None
        return router_logits

    def _compute_expert_outputs(
        self,
        flat_hidden_states: torch.Tensor,
        adapter_name: str,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = flat_hidden_states.size(0)
        final_hidden_states = torch.zeros(
            (num_tokens, self.output_features),
            dtype=flat_hidden_states.dtype,
            device=flat_hidden_states.device,
        )

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts[adapter_name]).permute(2, 1, 0)
        common_gate = self.base_layer.gate_proj(flat_hidden_states) # shared base projections for all experts, computed once for efficiency as explained in their paper
        common_up = self.base_layer.up_proj(flat_hidden_states)     # same here for up_proj
        act_fn = self._act_fns[adapter_name]

        for expert_idx in range(self.num_experts[adapter_name]):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue

            token_inputs = flat_hidden_states.index_select(0, top_x)
            gate_states = common_gate.index_select(0, top_x)
            up_states = common_up.index_select(0, top_x)

            gate_delta = self.experts.forward_delta("gate_proj", expert_idx, token_inputs, adapter_name)
            if gate_delta is not None:
                gate_states = gate_states + gate_delta.to(gate_states.dtype)

            up_delta = self.experts.forward_delta("up_proj", expert_idx, token_inputs, adapter_name)
            if up_delta is not None:
                up_states = up_states + up_delta.to(up_states.dtype)

            act_result = act_fn(gate_states) * up_states
            down_states = self.base_layer.down_proj(act_result)

            down_delta = self.experts.forward_delta("down_proj", expert_idx, act_result, adapter_name)
            if down_delta is not None:
                down_states = down_states + down_delta.to(down_states.dtype)

            weighted_states = down_states * routing_weights[top_x, idx, None].to(down_states.dtype)
            final_hidden_states.index_add_(0, top_x, weighted_states)   # aggregated weighted experts output

        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.disable_adapters or not self.active_adapters:
            return self.base_layer(hidden_states)

        if len(self.active_adapters) != 1:
            raise NotImplementedError()

        adapter_name = self.active_adapters[0]
        if adapter_name not in self.lora_router:
            return self.base_layer(hidden_states)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        router_inputs = hidden_states
        if self.training and self.jitter_noise[adapter_name] > 0:
            # adds routing noise to reduce expert collapse, but it's not specified in the original paper, only in the authors implementation
            jitter = torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise[adapter_name], 1.0 + self.jitter_noise[adapter_name]
            )
            router_inputs = router_inputs * jitter

        flat_hidden_states = hidden_states.reshape(-1, hidden_dim)
        flat_router_inputs = router_inputs.reshape(-1, hidden_dim)

        router = self.lora_router[adapter_name]
        router_logits = router(flat_router_inputs.to(router.weight.dtype))
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k[adapter_name], dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # compute and combine outputs from selected experts only
        mixed_states = self._compute_expert_outputs(
            flat_hidden_states=flat_hidden_states,
            adapter_name=adapter_name,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
        )

        self._latest_router_logits[adapter_name] = router_logits if self.router_loss[adapter_name] else None
        return mixed_states.reshape(batch_size, sequence_length, self.output_features).to(hidden_states.dtype)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("MixLoRA merge is not implemented.")

    def unmerge(self) -> None:
        raise NotImplementedError("MixLoRA merge is not implemented.")

    def extra_repr(self) -> str:
        adapter_names = sorted(self.lora_router.keys())
        return f"base_layer={type(self.base_layer).__name__}, projections={sorted(self.projection_shapes)}, adapters={adapter_names}"
