from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.metrics import record_grad_iso_metrics
from peft.tuners.tuners_utils import BaseTunerLayer


def _zero_touch_unused_partitions(layer, active_partitions: set, dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
    zero: Optional[torch.Tensor] = None
    for part_id in range(layer.num_partitions):
        if part_id in active_partitions:
            continue
        name = f"partition_{part_id}"
        for store_name in ("lora_A", "lora_B"):
            store = getattr(layer, store_name, None)
            if store is None or name not in store:
                continue
            module = store[name]
            for param in module.parameters():
                if not param.requires_grad or param.numel() == 0:
                    continue
                term = param.reshape(-1)[0].to(device=device, dtype=dtype)
                zero = term if zero is None else zero + term
    return None if zero is None else zero * 0.0


class GradIsoLoraLayer(BaseTunerLayer):
    _nonlayer_adapter_attrs = frozenset({"lora_A", "lora_B", "lora_dropout"})

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.lora_dropout = nn.ModuleDict({})
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.num_partitions = kwargs.get("num_partitions", 3)
        self.grad_iso_inference_mode = kwargs.get("grad_iso_inference_mode", "mean")
        self.language_guidance_scope = kwargs.get("language_guidance_scope", "all")
        self.language_list = kwargs.get("language_list")
        self.language_column = kwargs.get("language_column")
        self.track_router_metrics = kwargs.get("track_router_metrics", False)


class Linear(nn.Module, GradIsoLoraLayer):
    def __init__(self, base_layer: nn.Module, adapter_name: str, **kwargs):
        super().__init__()
        GradIsoLoraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, **kwargs)

    def update_layer(self, adapter_name: str, **kwargs):
        r = kwargs.get("r", 8)
        lora_alpha = kwargs.get("lora_alpha", 8)
        lora_dropout = kwargs.get("lora_dropout", 0.0)

        in_features = self.base_layer.in_features
        out_features = self.base_layer.out_features

        for part_id in range(self.num_partitions):
            name = f"partition_{part_id}"
            self.r[name] = r
            self.lora_alpha[name] = lora_alpha
            self.scaling[name] = lora_alpha / r

            self.lora_A[name] = nn.Linear(in_features, r, bias=False)
            self.lora_B[name] = nn.Linear(r, out_features, bias=False)

            nn.init.kaiming_uniform_(self.lora_A[name].weight, a=5**0.5)
            nn.init.zeros_(self.lora_B[name].weight)

        if lora_dropout > 0.0:
            self.lora_dropout["shared"] = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout["shared"] = nn.Identity()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        language_ids = kwargs.pop("language_ids", None)
        if language_ids is None:
            language_ids = getattr(self, "language_ids", None)

        result = self.base_layer(x, *args, **kwargs)
        result_dtype = result.dtype
        dropout = self.lora_dropout["shared"]

        if self.training and language_ids is not None:
            B, S, H = x.shape
            moe_out = torch.zeros_like(result)
            active_partitions = set()

            for part_id in range(self.num_partitions):
                mask = (language_ids == part_id)
                if not mask.any():
                    continue
                active_partitions.add(part_id)
                name = f"partition_{part_id}"
                x_sel = x[mask]
                h = self.lora_A[name](dropout(x_sel))
                out = self.lora_B[name](h) * self.scaling[name]
                moe_out[mask] = out.to(result_dtype)

            result = result + moe_out

            ddp_touch = _zero_touch_unused_partitions(self, active_partitions, dtype=result_dtype, device=result.device)
            if ddp_touch is not None:
                result = result + ddp_touch

            self._record_metrics(language_ids, active_partitions)
        else:
            partition_sum = torch.zeros_like(result)
            for part_id in range(self.num_partitions):
                name = f"partition_{part_id}"
                h = self.lora_A[name](x)
                out = self.lora_B[name](h) * self.scaling[name]
                partition_sum = partition_sum + out.to(result_dtype)
            result = result + partition_sum / self.num_partitions

        return result

    def _record_metrics(self, language_ids: torch.Tensor, active_partitions: set) -> None:
        if not self.track_router_metrics:
            return
        with torch.no_grad():
            metrics = {
                "partition_utilization": len(active_partitions) / self.num_partitions,
                "active_partitions": float(len(active_partitions)),
            }
            for part_id in range(self.num_partitions):
                count = (language_ids == part_id).sum().item()
                metrics[f"partition_{part_id}_samples"] = float(count)
            record_grad_iso_metrics(metrics, weight=1.0)


class Embedding(nn.Module, GradIsoLoraLayer):
    def __init__(self, base_layer: nn.Module, adapter_name: str, **kwargs):
        super().__init__()
        GradIsoLoraLayer.__init__(self, base_layer, **kwargs)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.base_layer(x, *args, **kwargs)
