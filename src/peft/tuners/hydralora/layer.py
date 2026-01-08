from __future__ import annotations

import math
import os
import warnings
import logging

logger = logging.getLogger(__name__)

from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose
from peft.tuners.utils.language_routing import (
    apply_language_bias,
    append_router_target_metrics,
    enforce_language_expert_routing,
    enforce_language_head_weights,
    language_expert_targets,
    language_head_targets,
)

from .forward import forward_flat as hydra_forward_flat
from .forward import forward_expert as hydra_forward_expert
from .config import HydraLoraConfig

import sys
from dataclasses import dataclass, fields


def debug(msg: str):
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)

_GLOBAL_DEBUG_ROUTING_EVERY = int(os.environ.get("HYDRA_DEBUG_ROUTING_EVERY", "0") or 0)


LANGUAGE_PAD_ID = -1


@dataclass
class HydraLayerConfig:
    use_hydralora_experts: bool = False
    hydralora_num_experts: int = 1
    hydralora_top_k: int = 1
    hydralora_head_top_k: Optional[int] = None
    hydralora_debug: bool = False
    language_list: Optional[list[str]] = None
    family_list: Optional[list[str]] = None
    language_to_family_ids: Optional[list[int]] = None
    language_to_subgroup_ids: Optional[list[int]] = None
    language_router_mode: str = "learned"
    language_bias_value: float = 0.0
    language_head_router_mode: Optional[str] = None
    language_head_bias_value: Optional[float] = None
    language_column: Optional[str] = None
    language_guidance_scope: str = "all"
    language_prior_weight: float = 0.0
    track_router_metrics: Optional[bool] = None
    hydralora_expert_lora_nums: Optional[list[int]] = None

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> tuple["HydraLayerConfig", dict[str, Any]]:
        data = {}
        for field in fields(cls):
            if field.name in kwargs:
                data[field.name] = kwargs.pop(field.name)
        return cls(**data), kwargs

    def __post_init__(self) -> None:
        if self.language_guidance_scope not in {"all", "expert_only", "none"}:
            raise ValueError(f"Unknown language_guidance_scope '{self.language_guidance_scope}'.")
        if self.language_head_router_mode is None:
            self.language_head_router_mode = self.language_router_mode
        if self.language_head_bias_value is None:
            self.language_head_bias_value = self.language_bias_value
        if self.hydralora_head_top_k is not None and self.hydralora_head_top_k <= 0:
            self.hydralora_head_top_k = None
        if self.track_router_metrics is None:
            self.track_router_metrics = self.language_prior_weight > 0 or self.language_router_mode == "hard"


class HydraLoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_route")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.lora_num = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.lora_route = nn.ModuleDict({})

        # Hierarchical / MoE bookkeeping
        self._hydra_expert_parent: dict[str, str] = {}
        self._hydra_parent_children: dict[str, list[str]] = {}

        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        setattr(self, "_active_adapters", [])
        layer_cfg, kwargs = HydraLayerConfig.from_kwargs(kwargs)
        expert_lora_nums_override = layer_cfg.hydralora_expert_lora_nums

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

        # self.W_V = None
        # self.W_S = None
        # self.Uh = None

        # hierarchical design addition
        self.use_hydralora_experts = layer_cfg.use_hydralora_experts
        self.num_experts = layer_cfg.hydralora_num_experts
        self.top_k = layer_cfg.hydralora_top_k
        self.head_top_k = layer_cfg.hydralora_head_top_k
        self.hydralora_debug = layer_cfg.hydralora_debug
        self.language_list = layer_cfg.language_list
        self.family_list = layer_cfg.family_list
        self.language_to_family_ids = layer_cfg.language_to_family_ids
        self.language_to_subgroup_ids = layer_cfg.language_to_subgroup_ids
        self.language_router_mode = layer_cfg.language_router_mode
        self.language_bias_value = layer_cfg.language_bias_value
        self.language_head_router_mode = layer_cfg.language_head_router_mode
        self.language_head_bias_value = layer_cfg.language_head_bias_value
        self.language_prior_weight = layer_cfg.language_prior_weight
        self.track_router_metrics = bool(layer_cfg.track_router_metrics)
        self.language_column = layer_cfg.language_column
        self.language_guidance_scope = layer_cfg.language_guidance_scope
        self._language_to_idx = {lang: idx for idx, lang in enumerate(self.language_list)} if self.language_list else {}
        if self.language_list:
            if self.language_to_family_ids is not None:
                expert_mapping = torch.tensor(self.language_to_family_ids, dtype=torch.long)
            else:
                expert_mapping = torch.arange(len(self.language_list), dtype=torch.long)
        else:
            expert_mapping = torch.empty(0, dtype=torch.long)
        self.register_buffer("language_id_to_expert", expert_mapping, persistent=False)
        family_mapping = (
            torch.tensor(self.language_to_family_ids, dtype=torch.long)
            if self.language_to_family_ids is not None
            else torch.empty(0, dtype=torch.long)
        )
        self.register_buffer("language_id_to_family", family_mapping, persistent=False)
        subgroup_mapping = (
            torch.tensor(self.language_to_subgroup_ids, dtype=torch.long)
            if self.language_list is not None and self.language_to_subgroup_ids is not None
            else torch.empty(0, dtype=torch.long)
        )
        self.register_buffer("language_id_to_subgroup", subgroup_mapping, persistent=False)

        if self.use_hydralora_experts:
            if self.family_list:
                self.num_experts = len(self.family_list)
            elif self.language_list:
                self.num_experts = len(self.language_list)
            if self.top_k > self.num_experts:
                self.top_k = self.num_experts
            self.router = nn.Linear(self.in_features, self.num_experts, bias=False)
            self._move_router_to_device_of_base_layer()
            if self.hydralora_debug:
                debug(
                    f"[HYDRA DEBUG] Initialized router in {self.__class__.__name__} "
                    f"(in={self.in_features}, experts={self.num_experts}, top_k={self.top_k})"
                )
        self._missing_language_warning_emitted: set[str] = set()
        self._expert_lora_nums = self._normalize_expert_counts(expert_lora_nums_override)
        self._debug_forward_calls = 0

    def _should_debug_routing(self) -> bool:
        if not getattr(self, "hydralora_debug", False):
            return False
        every = _GLOBAL_DEBUG_ROUTING_EVERY
        if every <= 0:
            return True
        self._debug_forward_calls += 1
        return (self._debug_forward_calls % every) == 0

    def _debug_routing_sample(
        self,
        x: torch.Tensor,
        language_ids: Optional[torch.Tensor],
        expert_targets: Optional[torch.Tensor],
        topi: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            debug(f"[HYDRA DEBUG] Router in {self.__class__.__name__}")
            sample_b, sample_t = 0, 0
            lang_id = int(language_ids[sample_b].item()) if language_ids is not None else LANGUAGE_PAD_ID
            tgt_ex = (
                int(expert_targets[sample_b].item())
                if expert_targets is not None and torch.is_tensor(expert_targets)
                else LANGUAGE_PAD_ID
            )
            sample_w = weights[sample_b, sample_t].detach().cpu().tolist()
            sample_i = topi[sample_b, sample_t].detach().cpu().tolist()
            debug(
                f"[HYDRA DEBUG] Sample(b={sample_b},t={sample_t}) lang_id={lang_id} "
                f"expert_target={tgt_ex} top_experts={sample_i}, weights={sample_w}"
            )
            if tgt_ex >= 0:
                ex_name = f"expert_{tgt_ex}"
                head_router = self.lora_route[ex_name] if ex_name in self.lora_route else None
                head_count = int(self.lora_num.get(ex_name, 0) or 0)
                if head_router is not None and head_count > 1:
                    head_logits = head_router(x.to(torch.float32)).to(x.dtype)
                    head_targets = (
                        self._language_head_targets(language_ids, ex_name)
                        if self.language_guidance_scope == "all"
                        else None
                    )
                    head_logits = self._apply_language_bias_heads(head_logits, head_targets)
                    head_probs = self._head_router_weights(head_logits)
                    head_probs_tok = head_probs[sample_b, sample_t].detach().cpu()
                    top_h = torch.topk(head_probs_tok, k=min(3, head_count), dim=-1)
                    debug(
                        f"[HYDRA DEBUG]   head_target="
                        f"{int(head_targets[sample_b].item()) if head_targets is not None else LANGUAGE_PAD_ID} "
                        f"top_heads={top_h.indices.tolist()} probs={top_h.values.tolist()}"
                    )

    def _debug_print_hydra_setup(self, parent: str) -> None:
        if not getattr(self, "hydralora_debug", False):
            return
        if not getattr(self, "use_hydralora_experts", False):
            return
        debug(f"[HYDRA DEBUG] Setup(parent='{parent}'): experts={self.num_experts}, top_k={self.top_k}")
        for e in range(self.num_experts):
            name = f"expert_{e}"
            head_cnt = int(self.lora_num.get(name, 0) or 0)
            langs: list[str] = []
            if self.language_list:
                if self.family_list and self.language_to_family_ids is not None:
                    for lang_idx, fam_idx in enumerate(self.language_to_family_ids):
                        if int(fam_idx) == e and lang_idx < len(self.language_list):
                            langs.append(self.language_list[lang_idx])
                else:
                    if e < len(self.language_list):
                        langs.append(self.language_list[e])
            lang_preview = ",".join(langs[:5])
            suffix = f", langs=[{lang_preview}{'...' if len(langs) > 5 else ''}]" if langs else ""
            debug(f"[HYDRA DEBUG]   {name}: lora_num={head_cnt}{suffix}")
        debug("=" * 60)

    def update_layer(
            self, adapter_name, r, lora_alpha, lora_dropout, lora_num, init_lora_weights
    ):
        # Hierarchical HydraLoRA
        if self.use_hydralora_experts and adapter_name == getattr(self, "_active_adapter", adapter_name):
            self._active_adapters = []
            adapter_children: list[str] = []

            for e in range(self.num_experts):
                name = f"expert_{e}"
                adapter_children.append(name)
                self._hydra_expert_parent[name] = adapter_name

                expert_lora_num = self._expert_lora_nums[e] if self._expert_lora_nums else lora_num
                self.r[name] = r
                self.lora_alpha[name] = lora_alpha
                self.lora_num[name] = expert_lora_num
                self.lora_dropout[name] = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
                # Core Hydra concept: one A, multiple B per expert
                self.lora_A[name] = nn.Linear(self.in_features, r, bias=False)
                self.lora_B[name] = nn.ModuleList(
                    [nn.Linear(r, self.out_features, bias=False) for _ in range(expert_lora_num)]
                )
                self.lora_route[name] = nn.Linear(self.in_features, expert_lora_num, bias=False)
                self.scaling[name] = lora_alpha / r

                self.reset_lora_parameters(name, init_lora_weights)

                self._move_adapter_to_device_of_base_layer(name)
                self._active_adapters.append(name)

            self._hydra_parent_children[adapter_name] = adapter_children

            if self.hydralora_debug:
                debug(
                    f"[HYDRA DEBUG] Created {self.num_experts} HydraLoRA experts for parent '{adapter_name}' "
                    f"(r={r}, lora_num={lora_num}, top_k={self.top_k})"
                )
                self._debug_print_hydra_setup(adapter_name)

            # Use the *parent* name for external API; children are stored in _hydra_parent_children
            self.set_adapter(adapter_name)
            return

        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.lora_num[adapter_name] = lora_num
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.ModuleList(
            [nn.Linear(r, self.out_features, bias=False) for _ in range(lora_num)])
        self.lora_route[adapter_name] = nn.Linear(self.in_features, lora_num, bias=False)

        self.scaling[adapter_name] = lora_alpha / r

        self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        self._active_adapters = [adapter_name]
        self.set_adapter(self._active_adapters)

    def set_adapter(self, adapter_names: Union[str, list[str]]) -> None:
        """
        Override BaseTunerLayer.set_adapter to support hierarchical Hydra experts:
        - if a parent adapter is given, expand to its expert children.
        - optionally enable/disable router grad based on whether any Hydra parent is active.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        expanded: list[str] = []
        any_hydra_parent_active = False

        for name in adapter_names:
            # Parent: expand to its children
            if self.use_hydralora_experts and name in self._hydra_parent_children:
                expanded.extend(self._hydra_parent_children[name])
                any_hydra_parent_active = True
            else:
                expanded.append(name)
                # Also mark if an individual expert belonging to a Hydra parent is used
                if self.use_hydralora_experts and name in self._hydra_expert_parent:
                    any_hydra_parent_active = True

        # Router grads only when Hydra hierarchy is actually in use
        if self.use_hydralora_experts and hasattr(self, "router"):
            self._move_router_to_device_of_base_layer()
            self.router.requires_grad_(any_hydra_parent_active)

        # Defer to BaseTunerLayer for actual bookkeeping
        super().set_adapter(expanded)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            for layer in self.lora_B[adapter_name]:
                nn.init.zeros_(layer.weight)

            nn.init.kaiming_uniform_(self.lora_route[adapter_name].weight, a=math.sqrt(5))
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key, None)
        return value

    def _move_router_to_device_of_base_layer(self) -> None:
        if not hasattr(self, "router") or self.router is None:
            return
        base_layer = self.get_base_layer()
        weight = getattr(base_layer, "weight", None)
        if weight is None:
            return
        self.router.to(weight.device, dtype=weight.dtype)

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self._active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self._active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _normalize_expert_counts(self, counts: Optional[list[int]]) -> Optional[list[int]]:
        if not self.use_hydralora_experts or counts is None:
            return None
        values = list(counts)
        if not values:
            return None
        if len(values) != self.num_experts:
            raise ValueError(f"hydralora_expert_lora_nums must provide {self.num_experts} entries, got {len(values)}.")
        if any(v <= 0 for v in values):
            raise ValueError("hydralora_expert_lora_nums entries must be positive integers.")
        return values

    def _language_head_targets(
            self, language_ids: Optional[torch.Tensor], adapter_name: str
    ) -> Optional[torch.Tensor]:
        return language_head_targets(
            language_ids,
            self.language_list,
            getattr(self, "language_id_to_subgroup", None),
            self.lora_num.get(adapter_name),
            self.language_guidance_scope,
            pad_id=LANGUAGE_PAD_ID,
        )

    def _language_expert_targets(self, language_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return language_expert_targets(
            language_ids,
            self.language_id_to_expert,
            self.language_guidance_scope,
            pad_id=LANGUAGE_PAD_ID,
        )

    def _log_missing_language_targets(self, prefix: str, reason: str) -> None:
        key = f"{prefix}:{reason}"
        if key in self._missing_language_warning_emitted:
            return
        column = self.language_column or "<unset>"
        logger.warning(
            "HydraLoRA layer '%s' missing %s routing metadata (%s). Verify dataset column '%s' and language_map.",
            self.__class__.__name__,
            prefix,
            reason,
            column,
        )
        self._missing_language_warning_emitted.add(key)

    def _append_target_metrics(
        self,
        metrics: dict[str, float],
        metrics_weight: float,
        prefix: str,
        target_tensor: Optional[torch.Tensor],
        selection: torch.Tensor,
        probs: torch.Tensor,
        language_ids: Optional[torch.Tensor],
        expect_targets: bool,
    ) -> float:
        return append_router_target_metrics(
            metrics,
            metrics_weight,
            prefix=prefix,
            target_tensor=target_tensor,
            selection=selection,
            probs=probs,
            language_ids=language_ids,
            expect_targets=expect_targets,
            on_missing=lambda reason: self._log_missing_language_targets(prefix, reason),
        )

    def _apply_language_bias_heads(
            self, logits: torch.Tensor, head_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return apply_language_bias(
            logits, head_ids, self.language_head_router_mode, self.language_head_bias_value
        )

    def _enforce_language_heads(
            self, weights: torch.Tensor, head_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return enforce_language_head_weights(weights, head_ids, self.language_head_router_mode)

    def _head_router_weights(self, logits: torch.Tensor) -> torch.Tensor:
        head_count = logits.size(-1)
        head_top_k = self.head_top_k
        if head_top_k is None or head_top_k <= 0 or head_top_k >= head_count:
            return torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        topv, topi = torch.topk(logits, head_top_k, dim=-1)
        weights = torch.softmax(topv.to(torch.float32), dim=-1).to(logits.dtype)
        sparse = torch.zeros_like(logits)
        sparse.scatter_(-1, topi, weights)
        return sparse

    def _apply_language_bias_experts(
            self, logits: torch.Tensor, expert_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return apply_language_bias(logits, expert_ids, self.language_router_mode, self.language_bias_value)

    def _enforce_language_experts(
            self, topi: torch.Tensor, weights: torch.Tensor, expert_ids: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return enforce_language_expert_routing(topi, weights, expert_ids, self.language_router_mode, self.top_k)

    def _cache_router_state(
            self, logits: torch.Tensor, language_ids: Optional[torch.Tensor], prefix: str,
            targets: Optional[torch.Tensor]
    ) -> None:
        self._cache_store(f"{prefix}_router_logits", logits)
        if language_ids is not None and torch.is_tensor(language_ids):
            self._cache_store(f"{prefix}_router_language_ids", language_ids)
        if targets is not None and torch.is_tensor(targets):
            self._cache_store(f"{prefix}_router_targets", targets)


class Linear(nn.Module, HydraLoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            base_layer,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            lora_num: int = 1,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            is_target_conv_1d_layer: bool = False,
            init_lora_weights: Union[bool, str] = True,
            **kwargs,
    ) -> None:
        super().__init__()
        HydraLoraLayer.__init__(self, base_layer, **kwargs)
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
            return hydra_forward_expert(self, x, *args, language_ids=language_ids, **kwargs)
        return hydra_forward_flat(self, x, *args, language_ids=language_ids, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter][0].weight.device
        dtype = self.lora_B[adapter][0].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        # weight_B = self.lora_B[adapter].weight
        weight_B = torch.sum(torch.stack([layer.weight for layer in self.lora_B[adapter]]), dim=0)

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            # self.lora_B[adapter].weight.data = weight_B.to(dtype)
            for i in range(self.num_B[adapter]):
                self.lora_B[adapter][i].weight.data = self.lora_B[adapter][i].weight.to(dtype)

        return output_tensor

    def _adapter_delta(
        self,
        x: torch.Tensor,
        name: str,
        *,
        language_ids: Optional[torch.Tensor] = None,
        expert_id: Optional[int] = None,
        expert_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the HydraLoRA delta for a single adapter/expert.

        - Flat Hydra: routed mixture over B heads (handled in Linear.forward, not here).
        - Expert Hydra: we optionally apply the same head routing per expert when lora_route[name] exists.
        """
        A = self.lora_A[name]  # nn.Linear
        B_list = self.lora_B[name]  # ModuleList[nn.Linear]
        drop = self.lora_dropout[name]
        scale = self.scaling[name]

        if not B_list:
            return torch.zeros_like(x, dtype=self.get_base_layer().weight.dtype)

        intermediate = drop(x.to(A.weight.dtype))  # [B, S, D_in]
        a_dot_x = A(intermediate)  # [B, S, r]

        lora_route = self.lora_route[name] if name in self.lora_route else None
        if lora_route is None or len(B_list) == 1:
            out = sum(B(a_dot_x) for B in B_list)
            return out * scale

        route_logits = lora_route(x.to(torch.float32)).to(x.dtype)  # [B, S, num_heads]
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
        route_weight = self._head_router_weights(route_logits)  # [B, S, H]
        route_weight = self._enforce_language_heads(route_weight, head_targets)

        out = 0
        for i, B in enumerate(B_list):
            out = out + torch.unsqueeze(route_weight[:, :, i], -1) * B(a_dot_x)

        return out * scale

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

    def pop_language_router_cache(self) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        caches: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        # Head routers for standard Hydra adapters
        head_keys = [
            key for key in list(self._caches.keys()) if key.startswith("hydra_head_") and key.endswith("_router_logits")
        ]
        for key in head_keys:
            logits = self._cache_pop(key)
            targets = self._cache_pop(key.replace("_router_logits", "_router_targets"))
            if logits is not None and targets is not None:
                caches.append(("hydra_head", logits, targets))
            self._cache_pop(key.replace("_router_logits", "_router_language_ids"))
        # Global expert router
        logits = self._cache_pop("hydra_expert_router_logits")
        targets = self._cache_pop("hydra_expert_router_targets")
        if logits is not None and targets is not None:
            caches.append(("hydra_expert", logits, targets))
        self._cache_pop("hydra_expert_router_language_ids")
        return caches


class Embedding(nn.Module, HydraLoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
            self,
            base_layer: nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: Union[bool, str] = True,
            **kwargs,
    ) -> None:
        super().__init__()
        HydraLoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
        self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self._active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        kwargs.pop("language_ids", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self._active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result = result + (after_A @ embedding_B) * scaling
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep




def dispatch_default(
        target: torch.nn.Module,
        adapter_name: str,
        lora_config: HydraLoraConfig,
        **kwargs,
) -> Optional[torch.nn.Module]:
    """Create a HydraLoRA adapter module for supported layer types."""
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
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
