from __future__ import annotations

import warnings
from contextlib import contextmanager
from itertools import chain
from typing import Any, Optional

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import get_quantization_config

from ..lora import LoraModel
from .config import MtlLoraConfig
from .layer import LinearMtlLoraLayer, MtlLoraLayer


class MtlLoraModel(LoraModel):
    prefix: str = "lora_"

    def __init__(self, model: nn.Module, config: MtlLoraConfig, adapter_name: str = "default") -> None:
        self._cached_language_ids = None
        self._cached_task_ids = None
        super().__init__(model, config, adapter_name)
        for module in self.model.modules():
            if isinstance(module, MtlLoraLayer):
                module.set_mtllora_parent(self)

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        self._cached_language_ids = kwargs.get("language_ids", None)
        self._cached_task_ids = kwargs.get("task_ids", None)
        try:
            with super()._enable_peft_forward_hooks(*args, **kwargs):
                yield
        finally:
            self._cached_language_ids = None
            self._cached_task_ids = None

    def get_task_ids(self, adapter_name: str, batch_size: int, device: torch.device) -> torch.Tensor:
        config = self.peft_config[adapter_name]
        task_ids = self._cached_language_ids if config.use_language_ids_as_task_ids else self._cached_task_ids
        if task_ids is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if not torch.is_tensor(task_ids):
            task_ids = torch.tensor(task_ids, device=device)
        else:
            task_ids = task_ids.to(device)

        if task_ids.dim() == 0:
            task_ids = task_ids.view(1)
        if task_ids.dim() > 1:
            task_ids = task_ids.view(task_ids.shape[0], -1)[:, 0]

        if task_ids.numel() == 1 and batch_size > 1:
            task_ids = task_ids.expand(batch_size)
        elif task_ids.numel() != batch_size:
            if task_ids.numel() > batch_size:
                task_ids = task_ids[:batch_size]
            else:
                pad = torch.zeros(batch_size - task_ids.numel(), dtype=task_ids.dtype, device=device)
                task_ids = torch.cat([task_ids, pad], dim=0)

        task_ids = task_ids.long()
        valid = (task_ids >= 0) & (task_ids < config.task_num)
        return torch.where(valid, task_ids, torch.zeros_like(task_ids))

    def _create_and_replace(
        self,
        mtllora_config: MtlLoraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        if current_key is None:
            raise ValueError("Current key shouldn't be `None`")

        pattern_keys = list(chain(mtllora_config.rank_pattern.keys(), mtllora_config.alpha_pattern.keys()))
        target_name_key = next((key for key in pattern_keys if current_key.endswith(key)), current_key)
        rank = mtllora_config.rank_pattern.get(target_name_key, mtllora_config.r)
        alpha = mtllora_config.alpha_pattern.get(target_name_key, mtllora_config.lora_alpha)

        layer_kwargs = {
            "r": rank,
            "lora_alpha": alpha,
            "lora_dropout": mtllora_config.lora_dropout,
            "init_lora_weights": mtllora_config.init_lora_weights,
            "use_rslora": mtllora_config.use_rslora,
            "task_num": mtllora_config.task_num,
            "num_up_projections": mtllora_config.num_up_projections,
            "temperature": mtllora_config.temperature,
            "lambda_format": mtllora_config.lambda_format,
        }

        new_module_kwargs = {
            **layer_kwargs,
            "fan_in_fan_out": mtllora_config.fan_in_fan_out,
            "ephemeral_gpu_offload": mtllora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        for quant_method in ("gptq", "aqlm", "awq"):
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                new_module_kwargs[f"{quant_method}_quantization_config"] = quantization_config

        if isinstance(target, MtlLoraLayer):
            target.update_layer(adapter_name, **layer_kwargs)
            target.set_mtllora_parent(self)
        else:
            new_module = self._create_new_module(mtllora_config, adapter_name, target, **new_module_kwargs)
            new_module.set_mtllora_parent(self)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        mtllora_config: MtlLoraConfig,
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
                    "Setting fan_in_fan_out to False.",
                    UserWarning,
                )
                kwargs["fan_in_fan_out"] = mtllora_config.fan_in_fan_out = False
            return LinearMtlLoraLayer(base_layer=target, adapter_name=adapter_name, **kwargs)

        raise ValueError(
            f"Target module {target} is not supported. Currently, only `torch.nn.Linear` layers can be adapted with MTL-LoRA."
        )
