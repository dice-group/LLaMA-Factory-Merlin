from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from ..hydralora.config import HydraLoraConfig
from ..hydralora.layer import Embedding as HydraEmbedding
from ..hydralora.layer import HydraLoraLayer, logger
from .forward import forward_expert as hala_forward_expert


class HalaLoraLayer(HydraLoraLayer):
    _VALID_EXECUTION_MODES = {"grouped_sparse_expert_dense_head"}
    _missing_language_warning_emitted_global: set[str] = set()

    def __init__(
        self,
        base_layer: nn.Module,
        ephemeral_gpu_offload: bool = False,
        hala_execution_mode: str = "grouped_sparse_expert_dense_head",
        **kwargs,
    ) -> None:
        self.hala_execution_mode = hala_execution_mode
        super().__init__(base_layer, ephemeral_gpu_offload=ephemeral_gpu_offload, **kwargs)
        if self.hala_execution_mode not in self._VALID_EXECUTION_MODES:
            raise ValueError(f"Unsupported hala_execution_mode={self.hala_execution_mode!r}.")
        if not getattr(self, "use_hydralora_experts", False):
            raise ValueError("HALA exploration branch requires expert routing.")
        if int(getattr(self, "top_k", 0) or 0) != 1:
            raise ValueError("HALA exploration branch requires sparse expert top_k=1.")
        if int(getattr(self, "head_top_k", 0) or 0) != 1:
            raise ValueError("HALA exploration branch requires sparse head head_top_k=1.")
        if getattr(self, "language_guidance_scope", None) != "all":
            raise ValueError("HALA requires language_guidance_scope='all'.")
        if float(getattr(self, "language_prior_weight", 0.0) or 0.0) <= 0.0:
            raise ValueError("HALA requires language_prior_weight > 0 for LPR supervision.")

    def _log_missing_language_targets(self, prefix: str, reason: str) -> None:
        key = f"{prefix}:{reason}"
        if key in self._missing_language_warning_emitted_global:
            return
        column = self.language_column or "<unset>"
        logger.warning(
            "HALA layer '%s' missing %s routing metadata (%s). Verify dataset column '%s' and language_map.",
            self.__class__.__name__,
            prefix,
            reason,
            column,
        )
        self._missing_language_warning_emitted_global.add(key)

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
        hala_execution_mode = kwargs.pop("hala_execution_mode", "grouped_sparse_expert_dense_head")
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
        if not getattr(self, "use_hydralora_experts", False):
            raise ValueError("HALA exploration branch only supports expert routing.")
        return hala_forward_expert(self, x, *args, language_ids=language_ids, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        raise ValueError("HALA adapters are routing-dependent and cannot be merged into base weights.")

    def unmerge(self) -> None:
        if not self.merged:
            return
        raise ValueError("HALA adapters are routing-dependent and cannot be unmerged from base weights.")

    def get_delta_weight(self, adapter) -> torch.Tensor:
        raise ValueError(
            f"HALA adapter '{adapter}' is routing-dependent and has no input-independent delta weight for merging."
        )

    def pop_language_router_cache(self) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        caches: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        head_keys = [
            key for key in list(self._caches.keys()) if key.startswith("hydra_head_") and key.endswith("_router_logits")
        ]
        for key in head_keys:
            logits = self._cache_pop(key)
            targets = self._cache_pop(key.replace("_router_logits", "_router_targets"))
            if logits is not None and targets is not None:
                caches.append(("hala_head", logits, targets))
            self._cache_pop(key.replace("_router_logits", "_router_language_ids"))

        logits = self._cache_pop("hydra_expert_router_logits")
        targets = self._cache_pop("hydra_expert_router_targets")
        if logits is not None and targets is not None:
            caches.append(("hala_expert", logits, targets))
        self._cache_pop("hydra_expert_router_language_ids")
        return caches

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
