from __future__ import annotations

from typing import cast

import torch.nn as nn

from ..lora import LoraModel
from .config import MixLoraConfig
from .layer import MixLoraMoeLayout


class MixLoraModel(LoraModel):
    prefix: str = "lora_"

    def __init__(self, model: nn.Module, config: MixLoraConfig, adapter_name: str = "default") -> None:
        self.__dict__["_mixlora_pending_layouts"] = {}
        super().__init__(model, config, adapter_name)
        self.mixlora_moe_layouts = nn.ModuleDict(self.__dict__.pop("_mixlora_pending_layouts"))

    def _register_moe_layout(self, current_key: str, parent: nn.Module, target_name: str, config: MixLoraConfig) -> None:
        mlp_path = current_key.rsplit(".", 1)[0]
        layout_key = mlp_path.replace(".", "__")
        pending = self.__dict__.setdefault("_mixlora_pending_layouts", {})
        if layout_key not in pending:
            pending[layout_key] = MixLoraMoeLayout(
                mlp_path=mlp_path,
                base_mlp_type=type(parent).__name__,
                num_experts=config.num_experts,
                top_k=config.top_k,
                routing_strategy=config.routing_strategy,
                router_init_range=config.router_init_range,
                jitter_noise=config.jitter_noise,
                router_loss=config.router_loss,
                router_aux_loss_coef=config.router_aux_loss_coef,
                act_fn=config.act_fn,
                expert_rank=config.expert_lora_r or config.r,
                expert_alpha=config.expert_lora_alpha or config.lora_alpha,
                expert_dropout=(
                    config.expert_lora_dropout if config.expert_lora_dropout is not None else config.lora_dropout
                ),
            )
        layout = cast(MixLoraMoeLayout, pending[layout_key])
        layout.register_projection(target_name)

    def _create_and_replace(
        self,
        mixlora_config: MixLoraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        if current_key is None:
            raise ValueError("Current key shouldn't be `None`")

        if mixlora_config.moe_target_modules and any(current_key.endswith(name) for name in mixlora_config.moe_target_modules):
            self._register_moe_layout(current_key, parent, target_name, mixlora_config)
            return

        super()._create_and_replace(mixlora_config, adapter_name, target, target_name, parent, current_key)
