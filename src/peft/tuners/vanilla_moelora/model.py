from __future__ import annotations

import re
from itertools import chain
from typing import Any, Optional

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import get_quantization_config

from ..lora import LoraModel
from .config import VanillaMoELoraConfig
from .layer import LinearVanillaMoELoraLayer, VanillaMoELoraLayer


def _load_balancing_loss(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    router_probs = torch.softmax(router_logits, dim=-1)
    _, selected_experts = torch.topk(router_probs, k=top_k, dim=-1)
    num_experts = router_logits.shape[-1]
    num_tokens = router_logits.shape[0]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
    expert_fractions = expert_counts.float() / float(num_tokens * top_k)
    expert_probs = router_probs.float().mean(dim=0)
    return num_experts * torch.sum(expert_fractions * expert_probs)


class VanillaMoELoraModel(LoraModel):
    prefix: str = "lora_"

    def _create_and_replace(
        self,
        vanilla_moelora_config: VanillaMoELoraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        if current_key is None:
            raise ValueError("Current key shouldn't be `None`")

        pattern_keys = list(chain(vanilla_moelora_config.rank_pattern.keys(), vanilla_moelora_config.alpha_pattern.keys()))
        target_name_key = next((key for key in pattern_keys if re.match(rf".*\.{key}$", current_key)), current_key)
        rank = vanilla_moelora_config.rank_pattern.get(target_name_key, vanilla_moelora_config.r)
        alpha = vanilla_moelora_config.alpha_pattern.get(target_name_key, vanilla_moelora_config.lora_alpha)

        layer_kwargs = {
            "lora_rank": rank,
            "lora_alpha": alpha,
            "lora_dropout": vanilla_moelora_config.lora_dropout,
            "init_lora_weights": vanilla_moelora_config.init_lora_weights,
            "num_experts": vanilla_moelora_config.num_experts,
            "top_k": vanilla_moelora_config.top_k,
            "router_init_range": vanilla_moelora_config.router_init_range,
            "router_aux_loss_coef": vanilla_moelora_config.router_aux_loss_coef,
            "use_rslora": vanilla_moelora_config.use_rslora,
        }
        new_module_kwargs = {
            **layer_kwargs,
            "fan_in_fan_out": vanilla_moelora_config.fan_in_fan_out,
            "use_dora": False,
            "ephemeral_gpu_offload": vanilla_moelora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        for quant_method in ("gptq", "aqlm", "awq"):
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                new_module_kwargs[f"{quant_method}_quantization_config"] = quantization_config

        if isinstance(target, VanillaMoELoraLayer):
            target.update_layer(adapter_name, **layer_kwargs)
        else:
            new_module = self._create_new_module(vanilla_moelora_config, adapter_name, target, **new_module_kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        vanilla_moelora_config: VanillaMoELoraConfig,
        adapter_name: str,
        target: nn.Module,
        **kwargs: Any,
    ) -> nn.Module:
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            return LinearVanillaMoELoraLayer(base_layer=target, adapter_name=adapter_name, **kwargs)

        raise ValueError(
            f"Target module {target} is not supported. Currently, only `torch.nn.Linear` layers can be adapted."
        )

    def get_aux_loss(self, adapter_name: str = "default") -> Optional[torch.Tensor]:
        if adapter_name not in self.peft_config:
            return None

        config = self.peft_config[adapter_name]
        if config.router_aux_loss_coef <= 0:
            return None

        losses = []
        for module in self.model.modules():
            if not isinstance(module, VanillaMoELoraLayer):
                continue
            router_logits = module.pop_router_logits(adapter_name)
            if router_logits is not None and router_logits.numel() > 0:
                losses.append(_load_balancing_loss(router_logits.float(), config.top_k))

        if not losses:
            return None
        return torch.stack(losses).mean()
