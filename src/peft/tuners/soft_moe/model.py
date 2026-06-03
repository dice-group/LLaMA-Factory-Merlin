from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer

from ..hydralora.config import HydraLoraConfig
from ..hydralora.model import HydraLoraModel
from .layer import Linear, Embedding


def dispatch_default(
    target: nn.Module,
    adapter_name: str,
    lora_config: HydraLoraConfig,
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


class SoftMoeModel(HydraLoraModel):
    prefix: str = "lora_"

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        kwargs["soft_moe_temperature"] = getattr(lora_config, "soft_moe_temperature", 1.0)
        dispatchers = []
        if lora_config._custom_modules:
            def dynamic_dispatch_func(target, adapter_name, lora_config, **kwargs):
                new_module = None
                if isinstance(target, BaseTunerLayer):
                    target_base_layer = target.get_base_layer()
                else:
                    target_base_layer = target
                for key, custom_cls in lora_config._custom_modules.items():
                    if isinstance(target_base_layer, key):
                        new_module = custom_cls(target, adapter_name, **kwargs)
                        break
                return new_module
            dispatchers.append(dynamic_dispatch_func)

        dispatchers.append(dispatch_default)

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:
                break
        if new_module is None:
            raise ValueError(
                f"Target module {target} is not supported. Currently only torch.nn.Linear and torch.nn.Embedding are supported."
            )
        return new_module
