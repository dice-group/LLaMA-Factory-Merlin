from __future__ import annotations

import re
import torch

from typing import Any
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
    get_quantization_config,
)

from ..lora import LoraModel
from .config import MovLoraConfig
from .layer import LinearMovLoraLayer, MovLoraLayer


class MovLoraModel(LoraModel):
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

        layer_kwargs = {
            # Unused by MoV layer implementation; kept for compatibility with the
            # existing LoraModel create/update call path.
            "lora_rank": movlora_config.r,
            "lora_alpha": movlora_config.lora_alpha,
            "lora_dropout": movlora_config.lora_dropout,
            "init_lora_weights": movlora_config.init_lora_weights,
            "num_experts": movlora_config.num_experts,
            "router_top_k": movlora_config.router_top_k,
            "router_temperature": movlora_config.router_temperature,
            "router_jitter_noise": movlora_config.router_jitter_noise,
            "router_bias": movlora_config.router_bias,
            "router_init_std": movlora_config.router_init_std,
            "router_ignore_padding_tokens": movlora_config.router_ignore_padding_tokens,
            "use_rslora": movlora_config.use_rslora,
            "is_feedforward": is_feedforward,
        }

        new_module_kwargs = {
            **layer_kwargs,
            "fan_in_fan_out": movlora_config.fan_in_fan_out,
            "use_dora": False,
            "ephemeral_gpu_offload": movlora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        for quant_method in ("gptq", "aqlm", "awq"):
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                new_module_kwargs[f"{quant_method}_quantization_config"] = quantization_config

        if isinstance(target, MovLoraLayer):
            target.update_layer(adapter_name, **layer_kwargs)
        else:
            new_module = self._create_new_module(movlora_config, adapter_name, target, **new_module_kwargs)
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
            return LinearMovLoraLayer(base_layer=target, adapter_name=adapter_name, **kwargs)

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
