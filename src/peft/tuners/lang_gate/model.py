from __future__ import annotations

import re
import warnings
from typing import Optional

import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists

from .config import LangGateConfig
from .layer import Linear, Embedding


def dispatch_default(
    target: nn.Module,
    adapter_name: str,
    lora_config: LangGateConfig,
    **kwargs,
) -> Optional[nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, nn.Linear):
        if kwargs.get("fan_in_fan_out", False):
            warnings.warn("fan_in_fan_out is set to True but target is nn.Linear. Setting to False.")
            kwargs["fan_in_fan_out"] = False
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        kwargs["fan_in_fan_out"] = True
        new_module = Linear(target, adapter_name, **kwargs)
    return new_module


class LangGateModel(BaseTuner):
    prefix: str = "lora_"

    def __init__(self, model, config, adapter_name, **kwargs):
        super().__init__(model, config, adapter_name, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise
            return getattr(self.model, name)

    def _prepare_adapter_config(self, peft_config, model_config):
        if peft_config.target_modules is None:
            peft_config.target_modules = "all-linear"
        return peft_config

    def _create_and_replace(self, lora_config, adapter_name, target, target_name, parent, current_key):
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "language_list": lora_config.language_list,
            "language_column": lora_config.language_column,
            "language_guidance_scope": lora_config.language_guidance_scope,
            "lang_gate_type": lora_config.lang_gate_type,
            "lang_gate_init": lora_config.lang_gate_init,
            "track_router_metrics": lora_config.track_router_metrics,
        }
        new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
        if new_module is not None:
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        new_module = dispatch_default(target, adapter_name, lora_config=lora_config, **kwargs)
        if new_module is None:
            raise ValueError(
                f"Target module {target} is not supported. Currently only torch.nn.Linear and torch.nn.Embedding are supported."
            )
        return new_module

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        if hasattr(child, "base_layer"):
            child = child.base_layer
        if hasattr(new_module, "base_layer"):
            if hasattr(child, "weight"):
                new_module.base_layer.weight = child.weight
                if hasattr(child, "bias"):
                    new_module.base_layer.bias = child.bias

    def _set_adapter_layers(self, enabled=True):
        pass

    def enable_adapter_layers(self) -> None:
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        self._set_adapter_layers(enabled=False)

    @staticmethod
    def _check_target_module_exists(peft_config, key: str) -> bool:
        return check_target_module_exists(peft_config, key)

    def _mark_only_adapters_as_trainable(self, model):
        for n, p in model.named_parameters():
            if "lora_" in n or "gates" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def set_adapter(self, adapter_name):
        pass
