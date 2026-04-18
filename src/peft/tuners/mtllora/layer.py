from __future__ import annotations

import math
import warnings
import weakref
from abc import ABC
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..lora import LoraLayer


class MtlLoraB(nn.Module):
    def __init__(self, num_up_projections: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_up_projections, out_features, rank))


class MtlLoraLayer(LoraLayer, ABC):
    adapter_layer_names = LoraLayer.adapter_layer_names + ("lora_lambdas", "lora_B_weights")
    other_param_names = LoraLayer.other_param_names + (
        "task_num",
        "num_up_projections",
        "temperature",
        "lambda_format",
    )

    def __init__(self, base_layer: nn.Module, **kwargs: Any) -> None:
        super().__init__(base_layer, **kwargs)
        self.lora_lambdas = nn.ParameterDict({})
        self.lora_B_weights = nn.ParameterDict({})
        self.task_num = {}
        self.num_up_projections = {}
        self.temperature = {}
        self.lambda_format = {}
        self._mtllora_parent_ref = None

    def set_mtllora_parent(self, parent_model: nn.Module) -> None:
        self._mtllora_parent_ref = weakref.ref(parent_model)

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: bool,
        use_rslora: bool,
        task_num: int,
        num_up_projections: int,
        temperature: float,
        lambda_format: str,
    ) -> None:
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but got {r}.")
        if task_num <= 0:
            raise ValueError(f"`task_num` must be positive, got {task_num}.")
        if num_up_projections <= 0:
            raise ValueError(f"`num_up_projections` must be positive, got {num_up_projections}.")
        if temperature <= 0:
            raise ValueError(f"`temperature` must be positive, got {temperature}.")
        if lambda_format not in {"full", "diagonal"}:
            raise ValueError("`lambda_format` must be either 'full' or 'diagonal'.")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.task_num[adapter_name] = task_num
        self.num_up_projections[adapter_name] = num_up_projections
        self.temperature[adapter_name] = temperature
        self.lambda_format[adapter_name] = lambda_format

        if lora_dropout > 0.0:
            self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout[adapter_name] = nn.Identity()

        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = MtlLoraB(num_up_projections, self.out_features, r)
        if lambda_format == "full":
            lambda_param = torch.zeros(task_num, r, r)
        else:
            lambda_param = torch.zeros(task_num, r)
        self.lora_lambdas[adapter_name] = nn.Parameter(lambda_param)
        self.lora_B_weights[adapter_name] = nn.Parameter(torch.empty(task_num, num_up_projections))

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        self.reset_mtllora_parameters(adapter_name, init_lora_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_mtllora_parameters(self, adapter_name: str, init_lora_weights: bool) -> None:
        if init_lora_weights is False:
            return
        if init_lora_weights is not True:
            raise ValueError(f"Unsupported MTL-LoRA initialization: {init_lora_weights!r}.")

        nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[adapter_name].weight)
        if self.lambda_format[adapter_name] == "full":
            eye = torch.eye(
                self.r[adapter_name],
                dtype=self.lora_lambdas[adapter_name].dtype,
                device=self.lora_lambdas[adapter_name].device,
            )
            self.lora_lambdas[adapter_name].data.copy_(
                eye.unsqueeze(0).expand(self.task_num[adapter_name], -1, -1)
            )
        else:
            nn.init.ones_(self.lora_lambdas[adapter_name])
        nn.init.kaiming_uniform_(self.lora_B_weights[adapter_name], a=math.sqrt(5))

    def _get_task_ids(self, adapter_name: str, batch_size: int, device: torch.device) -> torch.Tensor:
        parent_model = None if self._mtllora_parent_ref is None else self._mtllora_parent_ref()
        if parent_model is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        return parent_model.get_task_ids(
            adapter_name=adapter_name,
            batch_size=batch_size,
            device=device,
        )

    def _adapter_delta(self, x: torch.Tensor, adapter_name: str) -> torch.Tensor:
        squeeze_seq = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_seq = True
        if x.dim() != 3:
            raise ValueError(f"MTL-LoRA expects 2D or 3D linear inputs, got shape {tuple(x.shape)}.")

        batch_size = x.shape[0]
        task_ids = self._get_task_ids(adapter_name, batch_size, x.device)
        x_cast = x.to(self.lora_A[adapter_name].weight.dtype)
        after_a = self.lora_A[adapter_name](self.lora_dropout[adapter_name](x_cast))

        lambdas = self.lora_lambdas[adapter_name][task_ids]
        if self.lambda_format[adapter_name] == "diagonal":
            after_lambda = after_a * lambdas.unsqueeze(1).to(after_a.dtype)
        else:
            after_lambda = torch.bmm(after_a, lambdas.transpose(-2, -1).to(after_a.dtype))

        b_weights = F.softmax(
            self.lora_B_weights[adapter_name] / self.temperature[adapter_name],
            dim=-1,
            dtype=torch.float32,
        ).to(self.lora_B[adapter_name].weight.dtype)
        b_bank = self.lora_B[adapter_name].weight
        task_b = b_weights @ b_bank.reshape(self.num_up_projections[adapter_name], -1)
        task_b = task_b.reshape(self.task_num[adapter_name], self.out_features, self.r[adapter_name])
        b_mixed = task_b[task_ids]
        delta = torch.bmm(after_lambda, b_mixed.transpose(-2, -1)) * self.scaling[adapter_name]
        if squeeze_seq:
            delta = delta.squeeze(1)
        return delta


class LinearMtlLoraLayer(nn.Module, MtlLoraLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        task_num: int = 1,
        num_up_projections: int = 3,
        temperature: float = 0.1,
        lambda_format: str = "full",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        MtlLoraLayer.__init__(self, base_layer=base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        if fan_in_fan_out:
            warnings.warn(
                "fan_in_fan_out is set to True but MTL-LoRA currently supports torch.nn.Linear targets only. "
                "Setting fan_in_fan_out to False.",
                UserWarning,
            )
            self.fan_in_fan_out = False
        self.update_layer(
            adapter_name=adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            task_num=task_num,
            num_up_projections=num_up_projections,
            temperature=temperature,
            lambda_format=lambda_format,
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("MTL-LoRA adapters cannot be merged because deltas depend on task ids.")

    def unmerge(self) -> None:
        self.merged_adapters.clear()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)
        if self.disable_adapters:
            return result.to(previous_dtype)

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue
            delta = self._adapter_delta(x, active_adapter)
            result = result + delta.to(result.dtype)

        return result.to(previous_dtype)
