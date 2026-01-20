from __future__ import annotations

import re
from itertools import chain
from typing import Any

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import get_quantization_config

from ..lora import LoraModel
from .config import MolaConfig
from .layer import LinearMolaLayer, MolaLayer, TopKMoeLayer


class MolaModel(LoraModel):
    """
    MoLA (Mixture of LoRA Experts) Model implemented as a PEFT tuner.
    """

    prefix: str = "lora_"

    def __init__(self, model: nn.Module, config: MolaConfig, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)

    def _create_and_replace(
        self,
        mola_config: MolaConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        if current_key is None:
            raise ValueError("Current key shouldn't be `None`")

        pattern_keys = list(chain(mola_config.rank_pattern.keys(), mola_config.alpha_pattern.keys()))
        target_name_key = next(
            (key for key in pattern_keys if re.match(rf".*\.{key}$", current_key)),
            current_key,
        )
        rank = mola_config.rank_pattern.get(target_name_key, mola_config.r)
        alpha = mola_config.alpha_pattern.get(target_name_key, mola_config.lora_alpha)
        num_null_experts = mola_config.mola_num_null_experts
        if num_null_experts <= 0 and mola_config.mola_use_null_expert:
            num_null_experts = 1

        layer_kwargs = {
            "lora_rank": rank,
            "lora_alpha": alpha,
            "lora_dropout": mola_config.lora_dropout,
            "init_lora_weights": mola_config.init_lora_weights,
            "num_experts": mola_config.mola_num_experts,
            "top_k": mola_config.mola_top_k,
            "num_null_experts": num_null_experts,
            "output_router_logits": mola_config.mola_output_router_logits,
            "router_aux_loss_coef": mola_config.mola_router_aux_loss_coef,
            "aux_loss_annealing": mola_config.mola_aux_loss_annealing,
            "mola_debug_mode": mola_config.mola_debug_mode,
        }

        new_module_kwargs = {
            **layer_kwargs,
            "fan_in_fan_out": mola_config.fan_in_fan_out,
            "use_rslora": mola_config.use_rslora,
            "use_dora": mola_config.use_dora,
            "ephemeral_gpu_offload": mola_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        for quant_method in ("gptq", "aqlm", "awq"):
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                new_module_kwargs[f"{quant_method}_quantization_config"] = quantization_config

        if isinstance(target, MolaLayer):
            target.update_layer(adapter_name, **layer_kwargs)
        else:
            new_module = self._create_new_module(mola_config, adapter_name, target, **new_module_kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        mola_config: MolaConfig,
        adapter_name: str,
        target: nn.Module,
        **kwargs: Any,
    ) -> nn.Module:
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            return LinearMolaLayer(base_layer=target, adapter_name=adapter_name, **kwargs)

        raise ValueError(
            f"Target module {target} is not supported. "
            "Currently, only `torch.nn.Linear` layers can be adapted with MoLA."
        )

    @staticmethod
    def _extract_layer_key(name: str) -> str:
        for pattern in (r"\.layers\.(\d+)\.", r"\.layer\.(\d+)\.", r"\.h\.(\d+)\."):
            match = re.search(pattern, name)
            if match is not None:
                return match.group(1)
        return name

    def get_aux_loss(self, adapter_name: str = "default") -> torch.Tensor | None:
        param = next(self.model.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        model_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        layer_groups: dict[str, list[tuple[TopKMoeLayer, torch.Tensor]]] = {}

        for name, module in self.model.named_modules():
            if not name.endswith("moe_layer"):
                continue
            if isinstance(module, nn.ModuleDict) and adapter_name in module:
                layer = module[adapter_name]
            else:
                continue
            if not isinstance(layer, TopKMoeLayer):
                continue
            pop_fn = getattr(layer, "pop_router_state", None)
            if not callable(pop_fn):
                continue
            gate_logits = pop_fn()
            if gate_logits is None:
                continue
            layer_key = self._extract_layer_key(name)
            layer_groups.setdefault(layer_key, []).append((layer, gate_logits))

        any_loss = False
        for _, entries in layer_groups.items():
            ref_layer = entries[0][0]
            num_experts = entries[0][1].shape[-1]
            top_k = ref_layer.top_k
            gate_list = []
            for layer, gate_logits in entries:
                if gate_logits.shape[-1] != num_experts or layer.top_k != top_k:
                    continue
                gate_list.append(gate_logits)
            if not gate_list:
                continue
            gate_logits = torch.cat(gate_list, dim=0)
            gate_probs = torch.softmax(gate_logits, dim=-1)
            _, selected_experts = torch.topk(gate_probs, k=top_k, dim=-1)
            model_loss = model_loss + ref_layer.get_layer_loss(gate_logits, selected_experts)
            any_loss = True

        if not any_loss:
            return None
        return model_loss
