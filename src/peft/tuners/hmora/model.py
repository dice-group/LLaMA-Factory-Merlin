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
from .config import HMoRaConfig
from .layer import HMoRaLayer, LinearHMoRaLayer, TaskEncoder, TokenRouter


class HMoRaModel(LoraModel):
    prefix: str = "lora_"

    _QKV_MODULES = {"q_proj", "k_proj", "v_proj", "query_key_value"}
    _WI_MODULES = {"gate_proj", "up_proj", "dense_h_to_4h"}

    @staticmethod
    def _resolve_dtype(value: Any) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value
        text = str(value).replace("torch.", "").lower()
        mapping = {
            "float16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(text, torch.float32)

    @staticmethod
    def _required_task_embedding_count(task_ids: Optional[torch.Tensor]) -> int:
        if task_ids is None or task_ids.numel() == 0:
            return 1
        valid = task_ids.to(dtype=torch.long)
        valid = valid[valid >= 0]
        if valid.numel() == 0:
            return 1
        return int(valid.max().item()) + 1

    def __init__(self, model: nn.Module, config: HMoRaConfig, adapter_name: str = "default") -> None:
        # Delay ModuleDict registration until after `super().__init__` has initialized `nn.Module`.
        self.lora_router_pool: dict[str, dict[str, TokenRouter]] | nn.ModuleDict = {}
        self.lora_task_encoder: dict[str, TaskEncoder] | nn.ModuleDict = {}
        self._hmora_token_routers: dict[str, list[TokenRouter]] = {}
        self._hmora_task_routers: dict[str, list[nn.Module]] = {}
        self._hmora_router_use_count: dict[str, dict[str, int]] = {}
        self._hmora_max_layer = self._infer_max_layer(model)
        self._hmora_aux_attention_mask: Optional[torch.Tensor] = None
        self._hmora_cached_input_ids: Optional[torch.Tensor] = None
        self._hmora_cached_language_ids: Optional[torch.Tensor] = None
        self._hmora_runtime_task_input_ids: Optional[torch.Tensor] = None
        self._hmora_runtime_task_attention_mask: Optional[torch.Tensor] = None
        self._hmora_runtime_task_ids: Optional[torch.Tensor] = None
        super().__init__(model, config, adapter_name)

        if not isinstance(self.lora_router_pool, nn.ModuleDict):
            module_pool = nn.ModuleDict({})
            for name, router_dict in self.lora_router_pool.items():
                module_pool[name] = nn.ModuleDict(router_dict)
            self.lora_router_pool = module_pool
        if not isinstance(self.lora_task_encoder, nn.ModuleDict):
            self.lora_task_encoder = nn.ModuleDict(self.lora_task_encoder)

        for name in self.peft_config.keys():
            self._ensure_task_encoder(name)

        for module in self.model.modules():
            if isinstance(module, LinearHMoRaLayer):
                module.set_hmora_parent(self)

    @staticmethod
    def _infer_max_layer(model: nn.Module) -> int:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return 1
        for attr in ("num_hidden_layers", "num_layers", "n_layer"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                return max(value - 1, 1)
        return 1

    @staticmethod
    def _extract_layer_id(module_name: str) -> int:
        for pattern in (r"\.layers\.(\d+)\.", r"\.layer\.(\d+)\.", r"\.h\.(\d+)\."):
            match = re.search(pattern, module_name)
            if match:
                return int(match.group(1))
        return 0

    def _build_router_key(self, adapter_name: str, current_key: str, target_name: str, config: HMoRaConfig) -> str:
        layer_id = self._extract_layer_id(current_key)
        module_name = target_name.split(".")[-1]

        if config.share_router_for_qkv and module_name in self._QKV_MODULES:
            suffix = "qkv"
        elif config.share_router_for_w_i and module_name in self._WI_MODULES:
            suffix = "wi"
        else:
            suffix = module_name

        return f"layer_{layer_id}_{suffix}"

    def _ensure_router(
        self,
        adapter_name: str,
        router_key: str,
        in_features: int,
        layer_id: int,
        config: HMoRaConfig,
    ) -> tuple[str, bool]:
        if in_features <= 0:
            in_features = int(config.hidden_size or 0)
        if in_features <= 0:
            raise ValueError(f"HMoRA router cannot infer input size for key `{router_key}`.")

        if isinstance(self.lora_router_pool, nn.ModuleDict):
            if adapter_name not in self.lora_router_pool:
                self.lora_router_pool[adapter_name] = nn.ModuleDict({})
            router_bucket = self.lora_router_pool[adapter_name]
        else:
            if adapter_name not in self.lora_router_pool:
                self.lora_router_pool[adapter_name] = {}
            router_bucket = self.lora_router_pool[adapter_name]
        if adapter_name not in self._hmora_router_use_count:
            self._hmora_router_use_count[adapter_name] = {}
        if adapter_name not in self._hmora_token_routers:
            self._hmora_token_routers[adapter_name] = []
        if adapter_name not in self._hmora_task_routers:
            self._hmora_task_routers[adapter_name] = []

        if router_key not in router_bucket:
            router = TokenRouter(
                hidden_size=in_features,
                num_experts=config.num_experts,
                dropout=config.dropout,
                num_router_mlp_layers=config.num_router_mlp_layers,
                router_hidden_dim=config.router_hidden_dim,
                layer_id=layer_id,
                max_layer_id=self._hmora_max_layer,
                use_task_router=config.use_task_router,
                task_router_only=config.task_router_only,
                epsilon_alpha=config.epsilon_alpha,
                alpha_shift=config.alpha_shift,
                alpha_low_bound=config.alpha_low_bound,
                alpha_up_bound=config.alpha_up_bound,
                top_k_routing_strategy=config.top_k_routing_strategy,
                top_k=config.top_k,
                gamma_div_balance_t=config.gamma_div_balance_t,
                gamma_div_certain_t=config.gamma_div_certain_t,
                gamma_div_balance_s=config.gamma_div_balance_s,
                gamma_div_certain_s=config.gamma_div_certain_s,
                dtype=self._resolve_dtype(config.torch_dtype),
            )
            router_bucket[router_key] = router
            self._hmora_token_routers[adapter_name].append(router)
            if router.task_router is not None:
                self._hmora_task_routers[adapter_name].append(router.task_router)
            self._hmora_router_use_count[adapter_name][router_key] = 0

        use_count = self._hmora_router_use_count[adapter_name][router_key]
        self._hmora_router_use_count[adapter_name][router_key] = use_count + 1
        use_cache = use_count > 0
        return router_key, use_cache

    def _ensure_task_encoder(self, adapter_name: str) -> None:
        config = self.peft_config[adapter_name]
        if not config.use_task_router:
            return
        if adapter_name in self.lora_task_encoder:
            return

        input_embeddings = self.model.get_input_embeddings()
        if input_embeddings is None:
            return

        task_embedding = None
        if config.task_token_id is not None:
            task_id = int(config.task_token_id)
            if 0 <= task_id < input_embeddings.weight.shape[0]:
                task_embedding = input_embeddings.weight.data[task_id].detach().clone().to(
                    dtype=self._resolve_dtype(config.torch_dtype)
                )

        hidden_size = config.hidden_size or input_embeddings.weight.shape[-1]
        task_encoder = TaskEncoder(
            hidden_size=hidden_size,
            dropout=config.dropout,
            num_encoder_layer=config.num_encoder_layer,
            task_embedding=task_embedding,
            num_task_embeddings=config.num_task_embeddings,
        )
        self.lora_task_encoder[adapter_name] = task_encoder

    def set_runtime_task_inputs(
        self,
        *,
        task_input_ids: Optional[torch.Tensor],
        task_attention_mask: Optional[torch.Tensor],
        task_ids: Optional[torch.Tensor],
    ) -> None:
        self._hmora_runtime_task_input_ids = task_input_ids
        self._hmora_runtime_task_attention_mask = task_attention_mask
        self._hmora_runtime_task_ids = task_ids

    def clear_runtime_task_inputs(self) -> None:
        self._hmora_runtime_task_input_ids = None
        self._hmora_runtime_task_attention_mask = None
        self._hmora_runtime_task_ids = None

    def _task_inputs_for_adapter(
        self,
        adapter_name: str,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        config = self.peft_config[adapter_name]

        task_input_ids = self._hmora_runtime_task_input_ids
        task_attention_mask = self._hmora_runtime_task_attention_mask
        if task_input_ids is None or task_attention_mask is None:
            task_input_ids = self._hmora_cached_input_ids
            task_attention_mask = self._hmora_aux_attention_mask

        task_ids = self._hmora_runtime_task_ids
        if task_ids is None and config.use_language_ids_as_task_ids:
            task_ids = self._hmora_cached_language_ids

        return task_input_ids, task_attention_mask, task_ids

    def get_router(self, adapter_name: str, router_key: str) -> Optional[TokenRouter]:
        if adapter_name not in self.lora_router_pool:
            return None
        if router_key not in self.lora_router_pool[adapter_name]:
            return None
        return self.lora_router_pool[adapter_name][router_key]

    def _set_task_weight(self, adapter_name: str) -> None:
        config = self.peft_config[adapter_name]
        if not config.use_task_router:
            return
        if adapter_name not in self.lora_task_encoder:
            return
        task_input_ids, task_attention_mask, task_ids = self._task_inputs_for_adapter(adapter_name)
        if task_input_ids is None or task_attention_mask is None:
            return

        input_embeddings = self.model.get_input_embeddings()
        if input_embeddings is None:
            return

        hidden_states = input_embeddings(task_input_ids)
        task_encoder = self.lora_task_encoder[adapter_name]
        if config.use_language_ids_as_task_ids:
            next_size = self._required_task_embedding_count(task_ids)
            if next_size > task_encoder.task_embedding.num_embeddings:
                if self.training:
                    raise ValueError(
                        f"HMoRA requires {next_size} task embeddings from language_ids, "
                        f"but only {task_encoder.task_embedding.num_embeddings} were initialized."
                    )
                task_encoder.ensure_task_embedding_capacity(next_size)
            config.num_task_embeddings = max(config.num_task_embeddings, next_size)
        hidden_states = hidden_states.to(task_encoder.task_embedding.weight.dtype)
        task_embed = task_encoder(hidden_states, task_attention_mask, task_ids=task_ids)

        for task_router in self._hmora_task_routers.get(adapter_name, []):
            task_router(task_embed)

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        self._hmora_cached_input_ids = kwargs.get("input_ids", None)
        self._hmora_aux_attention_mask = kwargs.get("attention_mask", None)
        self._hmora_cached_language_ids = kwargs.get("language_ids", None)

        for adapter_name in self.active_adapters:
            if adapter_name in self.peft_config:
                self._set_task_weight(adapter_name)

        try:
            with super()._enable_peft_forward_hooks(*args, **kwargs):
                yield
        finally:
            self._hmora_cached_input_ids = None
            self._hmora_cached_language_ids = None
            self.clear_runtime_task_inputs()

    def clear_router_state(self, adapter_name: str = "default") -> None:
        for router in self._hmora_token_routers.get(adapter_name, []):
            router.clear()
        for task_router in self._hmora_task_routers.get(adapter_name, []):
            clear_fn = getattr(task_router, "clear", None)
            if callable(clear_fn):
                clear_fn()

    def get_aux_loss(self, adapter_name: str = "default", *, include_task_router: bool = True) -> Optional[torch.Tensor]:
        if adapter_name not in self.peft_config:
            return None

        config = self.peft_config[adapter_name]
        attention_mask = self._hmora_aux_attention_mask
        if attention_mask is None:
            return None

        aux_losses = []

        for router in self._hmora_token_routers.get(adapter_name, []):
            if config.use_load_balancing_loss:
                aux_losses.append(router.load_balancing_loss(attention_mask))
            elif config.use_div_loss:
                aux_losses.append(router.divergence_loss(attention_mask))

        if include_task_router and config.use_div_loss and not config.use_load_balancing_loss:
            for task_router in self._hmora_task_routers.get(adapter_name, []):
                aux_losses.append(task_router.divergence_loss())

        self.clear_router_state(adapter_name)
        self._hmora_aux_attention_mask = None

        if not aux_losses:
            return None
        return torch.stack(aux_losses, dim=0).sum()

    def get_task_router_aux_loss(
        self,
        *,
        task_input_ids: torch.Tensor,
        task_attention_mask: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        adapter_name: str = "default",
    ) -> Optional[torch.Tensor]:
        if adapter_name not in self.peft_config:
            return None

        config = self.peft_config[adapter_name]
        if not config.use_task_router or not config.use_div_loss or config.use_load_balancing_loss:
            return None
        if adapter_name not in self.lora_task_encoder:
            return None

        input_embeddings = self.model.get_input_embeddings()
        if input_embeddings is None:
            return None

        task_encoder = self.lora_task_encoder[adapter_name]
        if config.use_language_ids_as_task_ids:
            next_size = self._required_task_embedding_count(task_ids)
            if next_size > task_encoder.task_embedding.num_embeddings:
                if self.training:
                    raise ValueError(
                        f"HMoRA requires {next_size} task embeddings from language_ids, "
                        f"but only {task_encoder.task_embedding.num_embeddings} were initialized."
                    )
                task_encoder.ensure_task_embedding_capacity(next_size)
            config.num_task_embeddings = max(config.num_task_embeddings, next_size)

        hidden_states = input_embeddings(task_input_ids)
        hidden_states = hidden_states.to(task_encoder.task_embedding.weight.dtype)
        task_embed = task_encoder(hidden_states, task_attention_mask, task_ids=task_ids)

        aux_losses = []
        for task_router in self._hmora_task_routers.get(adapter_name, []):
            task_router(task_embed)
            aux_losses.append(task_router.divergence_loss())
            task_router.clear()

        if not aux_losses:
            return None
        return torch.stack(aux_losses, dim=0).sum()

    def _create_and_replace(
        self,
        hmora_config: HMoRaConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        if current_key is None:
            raise ValueError("Current key shouldn't be `None`")

        pattern_keys = list(chain(hmora_config.rank_pattern.keys(), hmora_config.alpha_pattern.keys()))
        target_name_key = next((key for key in pattern_keys if re.match(rf".*\.{key}$", current_key)), current_key)
        rank = hmora_config.rank_pattern.get(target_name_key, hmora_config.r)
        alpha = hmora_config.alpha_pattern.get(target_name_key, hmora_config.lora_alpha)

        module_leaf = target_name.split(".")[-1]
        module_lora_set = set(hmora_config.target_modules_lora or [])
        use_single_lora = module_leaf in module_lora_set

        router_key = None
        router_use_cache = False
        layer_id = self._extract_layer_id(current_key)
        if not use_single_lora:
            if isinstance(target, BaseTunerLayer):
                target_base_layer = target.get_base_layer()
            else:
                target_base_layer = target
            router_key = self._build_router_key(adapter_name, current_key, target_name, hmora_config)
            router_key, router_use_cache = self._ensure_router(
                adapter_name=adapter_name,
                router_key=router_key,
                in_features=getattr(target_base_layer, "in_features", hmora_config.hidden_size or 0),
                layer_id=layer_id,
                config=hmora_config,
            )

        layer_kwargs = {
            "lora_rank": rank,
            "lora_alpha": alpha,
            "lora_dropout": hmora_config.dropout,
            "init_lora_weights": hmora_config.init_lora_weights,
            "num_experts": hmora_config.num_experts,
            "use_hydra_lora": hmora_config.use_hydra_lora,
            "use_single_lora": use_single_lora,
            "router_key": router_key,
            "router_use_cache": router_use_cache,
        }

        new_module_kwargs = {
            **layer_kwargs,
            "fan_in_fan_out": hmora_config.fan_in_fan_out,
            "use_dora": False,
            "ephemeral_gpu_offload": hmora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        for quant_method in ("gptq", "aqlm", "awq"):
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                new_module_kwargs[f"{quant_method}_quantization_config"] = quantization_config

        if isinstance(target, HMoRaLayer):
            target.update_layer(adapter_name, **layer_kwargs)
            target.set_hmora_parent(self)
        else:
            new_module = self._create_new_module(hmora_config, adapter_name, target, **new_module_kwargs)
            new_module.set_hmora_parent(self)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

        self._ensure_task_encoder(adapter_name)

    @staticmethod
    def _create_new_module(
        hmora_config: HMoRaConfig,
        adapter_name: str,
        target: nn.Module,
        **kwargs: Any,
    ) -> nn.Module:
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            return LinearHMoRaLayer(base_layer=target, adapter_name=adapter_name, **kwargs)

        raise ValueError(
            f"Target module {target} is not supported. Currently, only `torch.nn.Linear` layers can be adapted with HMoRA."
        )
