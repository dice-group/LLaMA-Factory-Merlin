from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.metrics import record_hala_metrics
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from ..utils.language_routing import LANGUAGE_PAD_ID

from ..hydralora.config import HydraLoraConfig
from ..hydralora.layer import Embedding as HydraEmbedding
from ..hydralora.layer import HydraLoraLayer, transpose
from .forward import forward_expert as hala_forward_expert
from .forward import forward_flat as hala_forward_flat


class HalaLoraLayer(HydraLoraLayer):
    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, hala_execution_mode: str = "dense_expert_dense_head", **kwargs) -> None:
        self.hala_execution_mode = hala_execution_mode
        super().__init__(base_layer, ephemeral_gpu_offload=ephemeral_gpu_offload, **kwargs)
        if self.hala_execution_mode not in {"dense_expert_dense_head", "sparse_expert_dense_head"}:
            raise ValueError(f"Unsupported hala_execution_mode={self.hala_execution_mode!r}.")

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        lora_num,
        init_lora_weights,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ):
        if use_rslora or use_dora:
            raise ValueError("HALA does not support rsLoRA or DoRA adapters.")
        return HydraLoraLayer.update_layer(
            self,
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_num=lora_num,
            init_lora_weights=init_lora_weights,
        )


class Linear(nn.Module, HalaLoraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        lora_num: int = 1,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        hala_execution_mode = kwargs.pop("hala_execution_mode", "dense_expert_dense_head")
        super().__init__()
        HalaLoraLayer.__init__(self, base_layer, hala_execution_mode=hala_execution_mode, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_num=lora_num,
            init_lora_weights=init_lora_weights,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        language_ids = kwargs.pop("language_ids", None)
        if language_ids is None:
            language_ids = getattr(self, "language_ids", None)
        if isinstance(language_ids, torch.Tensor):
            language_ids = language_ids.to(x.device).long()
            if language_ids.dim() > 1:
                language_ids = language_ids.view(language_ids.size(0))
        else:
            language_ids = None
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if self.merged:
            return self.base_layer(x, *args, **kwargs)
        if getattr(self, "use_hydralora_experts", False):
            return hala_forward_expert(self, x, *args, language_ids=language_ids, **kwargs)
        return hala_forward_flat(self, x, *args, language_ids=language_ids, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        raise ValueError("HALA adapters are routing-dependent and cannot be merged into base weights.")

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        raise ValueError("HALA adapters are routing-dependent and cannot be unmerged from base weights.")

    def get_delta_weight(self, adapter) -> torch.Tensor:
        raise ValueError(
            f"HALA adapter '{adapter}' is routing-dependent and has no input-independent delta weight for merging."
        )

    def _adapter_delta(
        self,
        x: torch.Tensor,
        name: str,
        *,
        language_ids: Optional[torch.Tensor] = None,
        expert_id: Optional[int] = None,
        expert_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        A = self.lora_A[name]
        B_list = self.lora_B[name]
        drop = self.lora_dropout[name]
        scale = self.scaling[name]

        if not B_list:
            return torch.zeros_like(x, dtype=self.get_base_layer().weight.dtype)

        intermediate = drop(x.to(A.weight.dtype))
        a_dot_x = A(intermediate)

        lora_route = self.lora_route[name] if name in self.lora_route else None
        if lora_route is None or len(B_list) == 1:
            out = sum(B(a_dot_x) for B in B_list)
            return out * scale

        route_dtype = lora_route.weight.dtype
        route_logits = lora_route(x.to(route_dtype)).to(x.dtype)
        head_targets: Optional[torch.Tensor] = None
        use_head_guidance = self.language_guidance_scope == "all"
        if use_head_guidance and language_ids is not None:
            head_targets = self._language_head_targets(language_ids, name)
            if (
                head_targets is not None
                and expert_id is not None
                and expert_targets is not None
                and torch.is_tensor(expert_targets)
            ):
                mismatch = expert_targets != int(expert_id)
                if mismatch.any():
                    head_targets = head_targets.clone()
                    head_targets[mismatch] = LANGUAGE_PAD_ID

            self._cache_router_state(route_logits, language_ids, f"hydra_head_{name}", head_targets)

        route_logits = self._apply_language_bias_heads(route_logits, head_targets)
        route_weight = self._head_router_weights(route_logits)
        route_weight = self._enforce_language_heads(route_weight, head_targets)
        if self.track_router_metrics:
            with torch.no_grad():
                token_count = route_weight.numel() // route_weight.size(-1)
                if token_count > 0:
                    head_assign = torch.argmax(route_weight, dim=-1, keepdim=True)
                    flat_assign = head_assign.reshape(-1)
                    head_counts = torch.bincount(flat_assign, minlength=len(B_list)).to(torch.float32)
                    mean_head = head_counts.mean().item()
                    head_cv = float((head_counts.std(unbiased=False) / (mean_head + 1e-6)).item()) if mean_head > 0 else 0.0
                    head_active = float((head_counts > 0).float().mean().item())
                    head_entropy = float((-route_weight * torch.log(route_weight + 1e-8)).sum(dim=-1).mean().item())
                    total_head_assign = head_counts.sum()
                    if total_head_assign > 0:
                        head_frac = head_counts / total_head_assign
                        head_max = float(head_frac.max().item())
                        head_min = float(head_frac.min().item())
                    else:
                        head_max = 0.0
                        head_min = 0.0
                    metrics = {
                        "head_load_cv": head_cv,
                        "head_active_frac": head_active,
                        "head_router_entropy": head_entropy,
                        "head_load_max_frac": head_max,
                        "head_load_min_frac": head_min,
                    }
                    metrics_weight = float(token_count)
                    coverage_metrics, target_metrics, target_weight = self._append_target_metrics(
                        metrics=metrics,
                        metrics_weight=metrics_weight,
                        prefix="head",
                        target_tensor=head_targets,
                        selection=head_assign.squeeze(-1),
                        probs=route_weight,
                        language_ids=language_ids,
                        expect_targets=use_head_guidance and self.language_list is not None,
                    )
                    metrics.update(coverage_metrics)
                    record_hala_metrics(metrics, weight=metrics_weight)
                    if target_metrics and target_weight > 0:
                        record_hala_metrics(target_metrics, weight=target_weight)

        out = 0
        for i, B in enumerate(B_list):
            out = out + torch.unsqueeze(route_weight[:, :, i], -1) * B(a_dot_x)

        return out * scale

    def __repr__(self) -> str:
        return "lora." + super().__repr__()


class Embedding(HydraEmbedding):
    pass


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: HydraLoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
