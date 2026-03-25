from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from peft.tuners.tuners_utils import _get_submodules, check_target_module_exists
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

from ..lora import LoraModel
from .config import MixLoraConfig
from .layer import MixLoraMoeLayer


class MixLoraModel(LoraModel):
    prefix: str = "lora_"

    def __init__(self, model: nn.Module, config: MixLoraConfig, adapter_name: str = "default") -> None:
        self.__dict__["_mixlora_replaced_mlps"] = set()
        super().__init__(model, config, adapter_name)

    def _check_target_module_exists(self, mixlora_config: MixLoraConfig, key: str):
        if mixlora_config.moe_target_modules and any(key.endswith(name) for name in mixlora_config.moe_target_modules):
            mlp_path = key.rsplit(".", 1)[0]
            if mlp_path in self.__dict__.setdefault("_mixlora_replaced_mlps", set()):
                return False
        return check_target_module_exists(mixlora_config, key)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)

        if hasattr(child, "base_layer"):
            child = child.base_layer

        meta = torch.device("meta")
        for name, module in new_module.named_modules():
            if self.prefix not in name:
                continue

            if hasattr(child, "qweight"):
                weight = child.qweight
            elif hasattr(child, "W_q"):
                weight = child.W_q
            elif hasattr(child, "weight"):
                weight = child.weight
            elif hasattr(child, "gate_proj"):
                weight = child.gate_proj.weight
            else:
                weight = next(child.parameters())

            if not any(param.device == meta for param in module.parameters()):
                module.to(weight.device)

    def _create_mixlora_mlp(
        self,
        mixlora_config: MixLoraConfig,
        adapter_name: str,
        current_key: str,
    ) -> None:
        mlp_path = current_key.rsplit(".", 1)[0]
        replaced_mlps = self.__dict__.setdefault("_mixlora_replaced_mlps", set())
        if mlp_path in replaced_mlps:
            return

        mlp_parent, mlp_target, mlp_target_name = _get_submodules(self.model, mlp_path)
        if isinstance(mlp_target, MixLoraMoeLayer):
            mlp_target.update_layer(adapter_name, mixlora_config)
        else:
            new_module = MixLoraMoeLayer(mlp_target, adapter_name=adapter_name, mixlora_config=mixlora_config)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(mlp_parent, mlp_target_name, new_module, mlp_target)

        replaced_mlps.add(mlp_path)

    def _create_and_replace(
        self,
        mixlora_config: MixLoraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        *,
        parameter_name: Optional[str] = None,
    ) -> None:
        del target
        del target_name
        del parent
        del parameter_name

        if current_key is None:
            raise ValueError("Current key shouldn't be `None`")

        if mixlora_config.moe_target_modules and any(current_key.endswith(name) for name in mixlora_config.moe_target_modules):
            self._create_mixlora_mlp(mixlora_config, adapter_name, current_key)
            return

        parent, target, target_name = _get_submodules(self.model, current_key)
        super()._create_and_replace(mixlora_config, adapter_name, target, target_name, parent, current_key)

    def get_aux_loss(self, adapter_name: str = "default") -> Optional[torch.Tensor]:
        if adapter_name not in self.peft_config:
            return None

        config = self.peft_config[adapter_name]
        if not getattr(config, "router_loss", False):
            return None

        all_router_logits = []
        for module in self.model.modules():
            if not isinstance(module, MixLoraMoeLayer):
                continue
            router_logits = module.pop_router_logits(adapter_name)
            if router_logits is not None:
                all_router_logits.append(router_logits)

        if not all_router_logits:
            return None

        aux = load_balancing_loss_func(tuple(all_router_logits), num_experts=config.num_experts, top_k=config.top_k)
        if isinstance(aux, int):
            return None
        return aux
