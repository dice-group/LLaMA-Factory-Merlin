from __future__ import annotations

import re
import warnings
import torch

from typing import Any
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
    get_quantization_config,
)
from transformers.pytorch_utils import Conv1D

from ..ia3 import IA3Layer, IA3Model
from .config import MovLoraConfig
from .layer import LinearMovLoraLayer, MovLoraLayer


class MovLoraModel(IA3Model):
    prefix: str = "lora_"

    def __init__(self, model: nn.Module, config: MovLoraConfig, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)

    def _create_and_replace(
        self,
        movlora_config: MovLoraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        if current_key is None:
            raise ValueError("Current key shouldn't be `None`")

        is_feedforward = self._check_target_module_feedforward(movlora_config, current_key)

        kwargs = {
            "fan_in_fan_out": movlora_config.fan_in_fan_out,
            "init_ia3_weights": movlora_config.init_ia3_weights,
            "is_feedforward": is_feedforward,
            "num_experts": movlora_config.num_experts,
            "router_top_k": movlora_config.router_top_k,
            "router_temperature": movlora_config.router_temperature,
            "router_jitter_noise": movlora_config.router_jitter_noise,
            "router_bias": movlora_config.router_bias,
            "router_init_std": movlora_config.router_init_std,
            "router_ignore_padding_tokens": movlora_config.router_ignore_padding_tokens,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        for quant_method in ("gptq", "aqlm", "awq"):
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        if isinstance(target, MovLoraLayer):
            target.update_layer(
                adapter_name=adapter_name,
                init_ia3_weights=movlora_config.init_ia3_weights,
                num_experts=movlora_config.num_experts,
                router_top_k=movlora_config.router_top_k,
                router_temperature=movlora_config.router_temperature,
                router_jitter_noise=movlora_config.router_jitter_noise,
                router_bias=movlora_config.router_bias,
                router_init_std=movlora_config.router_init_std,
                router_ignore_padding_tokens=movlora_config.router_ignore_padding_tokens,
                is_feedforward=is_feedforward,
            )
        elif isinstance(target, IA3Layer):
            raise ValueError("Cannot reuse IA3 layers with MoV adapters.")
        else:
            new_module = self._create_new_module(movlora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        movlora_config: MovLoraConfig,
        adapter_name: str,
        target: nn.Module,
        **kwargs: Any,
    ) -> nn.Module:
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs.get("fan_in_fan_out", False):
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = movlora_config.fan_in_fan_out = False
            return LinearMovLoraLayer(base_layer=target, adapter_name=adapter_name, **kwargs)
        elif isinstance(target_base_layer, Conv1D):
            raise ValueError("Conv1D targets are not supported for MoV in this implementation.")

        raise ValueError(
            f"Target module {target} is not supported. Currently, only `torch.nn.Linear` layers can be adapted with MoV-LoRA."
        )

    @staticmethod
    def _check_target_module_feedforward(movlora_config: MovLoraConfig, key: str) -> bool:
        if isinstance(movlora_config.feedforward_modules, str):
            return bool(re.fullmatch(movlora_config.feedforward_modules, key))

        return any(key.endswith(target_key) for target_key in movlora_config.feedforward_modules)

    @staticmethod
    def _prepare_adapter_config(peft_config: MovLoraConfig, model_config: dict) -> MovLoraConfig:
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING[model_config["model_type"]])

        if peft_config.feedforward_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING:
                raise ValueError("Please specify `feedforward_modules` in `peft_config`")
            peft_config.feedforward_modules = set(
                TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING[model_config["model_type"]]
            )

        return peft_config
