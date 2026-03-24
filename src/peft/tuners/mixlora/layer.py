from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn


@dataclass(frozen=True)
class MixLoraExpertShape:
    target_module: str
    rank: int
    alpha: int
    dropout: float


class MixLoraMoeLayout(nn.Module):
    def __init__(
        self,
        *,
        mlp_path: str,
        base_mlp_type: str,
        num_experts: int,
        top_k: int,
        routing_strategy: str,
        router_init_range: float,
        jitter_noise: float,
        router_loss: bool,
        router_aux_loss_coef: float,
        act_fn: Optional[str],
        expert_rank: int,
        expert_alpha: int,
        expert_dropout: float,
    ) -> None:
        super().__init__()
        self.mlp_path = mlp_path
        self.base_mlp_type = base_mlp_type
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_strategy = routing_strategy
        self.router_init_range = router_init_range
        self.jitter_noise = jitter_noise
        self.router_loss = router_loss
        self.router_aux_loss_coef = router_aux_loss_coef
        self.act_fn = act_fn
        self.expert_rank = expert_rank
        self.expert_alpha = expert_alpha
        self.expert_dropout = expert_dropout
        self.expert_shapes: dict[str, MixLoraExpertShape] = {}

    def register_projection(self, target_name: str) -> None:
        if target_name in self.expert_shapes:
            return
        self.expert_shapes[target_name] = MixLoraExpertShape(
            target_module=target_name,
            rank=self.expert_rank,
            alpha=self.expert_alpha,
            dropout=self.expert_dropout,
        )

    def extra_repr(self) -> str:
        return (
            f"path={self.mlp_path}, experts={self.num_experts}, top_k={self.top_k}, "
            f"routing={self.routing_strategy}, mlp_type={self.base_mlp_type}, "
            f"projections={sorted(self.expert_shapes)}"
        )
