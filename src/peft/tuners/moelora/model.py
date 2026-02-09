from __future__ import annotations

import re
from contextlib import contextmanager
from itertools import chain
from typing import Any, Optional

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import get_quantization_config

from ..lora import LoraModel
from .config import MoELoraConfig
from .layer import LinearMoELoraLayer, MoELoraLayer


class MoELoraModel(LoraModel):
    prefix: str = "lora_"

    def __init__(self, model: nn.Module, config: MoELoraConfig, adapter_name: str = "default") -> None:
        self.lora_task_embedding = None
        self.lora_gate = None
        self._cached_task_ids = None
        super().__init__(model, config, adapter_name)
        self.lora_task_embedding = nn.ModuleDict({})
        self.lora_gate = nn.ModuleDict({})
        for name in self.peft_config.keys():
            self._ensure_shared_gate(name)
        for module in self.model.modules():
            if isinstance(module, MoELoraLayer):
                module.set_moelora_parent(self)

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        self._cached_task_ids = kwargs.get("task_ids", None)
        try:
            with super()._enable_peft_forward_hooks(*args, **kwargs):
                yield
        finally:
            self._cached_task_ids = None

    def _ensure_shared_gate(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
        if self.lora_task_embedding is None or self.lora_gate is None:
            return
        if adapter_name in self.lora_task_embedding and adapter_name in self.lora_gate:
            return

        config = self.peft_config[adapter_name]
        self.lora_task_embedding[adapter_name] = nn.Embedding(config.task_num, config.task_embedding_dim)
        self.lora_gate[adapter_name] = nn.Linear(config.task_embedding_dim, config.expert_num, bias=False)
        nn.init.normal_(self.lora_task_embedding[adapter_name].weight, std=0.02)
        nn.init.xavier_uniform_(self.lora_gate[adapter_name].weight)

        if device is None:
            base_param = next(self.model.parameters(), None)
            device = base_param.device if base_param is not None else None
        if device is not None:
            self.lora_task_embedding[adapter_name].to(device)
            self.lora_gate[adapter_name].to(device)

    def _sanitize_task_ids(self, task_ids: Any, batch_size: int, task_num: int, device: torch.device) -> torch.Tensor:
        if task_ids is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if not torch.is_tensor(task_ids):
            task_ids = torch.tensor(task_ids, device=device)
        else:
            task_ids = task_ids.to(device)

        if task_ids.dim() == 0:
            task_ids = task_ids.view(1)

        if task_ids.dim() == 1:
            if task_ids.numel() == 1 and batch_size > 1:
                task_ids = task_ids.expand(batch_size)
            elif task_ids.numel() != batch_size:
                if task_ids.numel() > batch_size:
                    task_ids = task_ids[:batch_size]
                else:
                    task_ids = torch.cat(
                        [task_ids, torch.zeros(batch_size - task_ids.numel(), dtype=task_ids.dtype, device=device)], dim=0
                    )
            return torch.remainder(task_ids.long(), task_num)

        # Multi-task weights: [batch_size, task_num]
        if task_ids.shape[0] == 1 and batch_size > 1:
            task_ids = task_ids.expand(batch_size, -1)
        elif task_ids.shape[0] != batch_size:
            if task_ids.shape[0] > batch_size:
                task_ids = task_ids[:batch_size]
            else:
                pad = torch.zeros(batch_size - task_ids.shape[0], task_ids.shape[1], device=device, dtype=task_ids.dtype)
                task_ids = torch.cat([task_ids, pad], dim=0)

        if task_ids.shape[1] < task_num:
            pad = torch.zeros(batch_size, task_num - task_ids.shape[1], device=device, dtype=task_ids.dtype)
            task_ids = torch.cat([task_ids, pad], dim=1)
        elif task_ids.shape[1] > task_num:
            task_ids = task_ids[:, :task_num]

        return task_ids

    def get_routing_weights(
        self,
        adapter_name: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        config = self.peft_config[adapter_name]
        task_ids = self._sanitize_task_ids(self._cached_task_ids, batch_size, config.task_num, device)
        task_embedder = self.lora_task_embedding[adapter_name]
        gate = self.lora_gate[adapter_name]

        if task_ids.dim() == 1:
            task_embeds = task_embedder(task_ids)
        else:
            task_weights = task_ids.float()
            denom = task_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            task_weights = task_weights / denom
            task_embeds = task_weights @ task_embedder.weight

        logits = gate(task_embeds) / config.gate_temperature
        if config.gate_top_k and 0 < config.gate_top_k < config.expert_num:
            _, topk_idx = torch.topk(logits, k=config.gate_top_k, dim=-1)
            sparse_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
            sparse_logits.scatter_(1, topk_idx, logits.gather(1, topk_idx))
            logits = sparse_logits

        return torch.softmax(logits, dim=-1).to(dtype)

    def _create_and_replace(
        self,
        moelora_config: MoELoraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        if current_key is None:
            raise ValueError("Current key shouldn't be `None`")

        if self.lora_task_embedding is not None and self.lora_gate is not None:
            self._ensure_shared_gate(adapter_name)
        pattern_keys = list(chain(moelora_config.rank_pattern.keys(), moelora_config.alpha_pattern.keys()))
        target_name_key = next((key for key in pattern_keys if re.match(rf".*\.{key}$", current_key)), current_key)
        rank = moelora_config.rank_pattern.get(target_name_key, moelora_config.r)
        alpha = moelora_config.alpha_pattern.get(target_name_key, moelora_config.lora_alpha)

        layer_kwargs = {
            "lora_rank": rank,
            "lora_alpha": alpha,
            "lora_dropout": moelora_config.lora_dropout,
            "init_lora_weights": moelora_config.init_lora_weights,
            "expert_num": moelora_config.expert_num,
            "use_rslora": moelora_config.use_rslora,
        }

        new_module_kwargs = {
            **layer_kwargs,
            "fan_in_fan_out": moelora_config.fan_in_fan_out,
            "use_dora": False,
            "ephemeral_gpu_offload": moelora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        for quant_method in ("gptq", "aqlm", "awq"):
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                new_module_kwargs[f"{quant_method}_quantization_config"] = quantization_config

        if isinstance(target, MoELoraLayer):
            target.update_layer(adapter_name, **layer_kwargs)
            target.set_moelora_parent(self)
        else:
            new_module = self._create_new_module(moelora_config, adapter_name, target, **new_module_kwargs)
            new_module.set_moelora_parent(self)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        moelora_config: MoELoraConfig,
        adapter_name: str,
        target: nn.Module,
        **kwargs: Any,
    ) -> nn.Module:
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            return LinearMoELoraLayer(base_layer=target, adapter_name=adapter_name, **kwargs)

        raise ValueError(
            f"Target module {target} is not supported. Currently, only `torch.nn.Linear` layers can be adapted with MoE-LoRA."
        )
