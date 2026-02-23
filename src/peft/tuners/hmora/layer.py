from __future__ import annotations

import math
import weakref
from abc import ABC
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..lora import LoraLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, -x.size(1) :, :]


class TaskEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, num_encoder_layer: int, task_embedding: Optional[torch.Tensor]):
        super().__init__()
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead=16, dim_feedforward=hidden_size * 2, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layer)
        dtype = task_embedding.dtype if task_embedding is not None else torch.float32
        self.task_embedding = nn.Embedding(1, hidden_size, dtype=dtype)
        if task_embedding is not None:
            self.task_embedding.weight.data.copy_(task_embedding)

    def forward(self, src: torch.Tensor, src_attention_mask: torch.Tensor) -> torch.Tensor:
        task_embedding = self.task_embedding(torch.tensor([0], device=src.device)).expand(src.shape[0], -1, -1)
        src = self.pos_encoder(src)
        src = torch.cat([src, task_embedding], dim=1)
        src = src.transpose(0, 1)

        src_key_padding_mask = torch.cat(
            [
                src_attention_mask,
                torch.ones(src_attention_mask.shape[0], 1, device=src_attention_mask.device, dtype=src_attention_mask.dtype),
            ],
            dim=1,
        )
        src_key_padding_mask = src_key_padding_mask == 0
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        sentence_embedding = output[-1]
        return sentence_embedding


class TaskRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, dropout: float, num_router_mlp_layers: int, router_hidden_dim: int, gamma_div_balance: float, gamma_div_certain: float, dtype: torch.dtype):
        super().__init__()
        self.num_experts = num_experts
        self.gamma_div_balance = gamma_div_balance
        self.gamma_div_certain = gamma_div_certain

        if num_router_mlp_layers == 1:
            self.mlp = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_experts, dtype=dtype),
            )
        else:
            layers: list[nn.Module] = [
                nn.Dropout(dropout),
                nn.Linear(hidden_size, router_hidden_dim, dtype=dtype),
                nn.ReLU(),
            ]
            for _ in range(num_router_mlp_layers - 2):
                layers.extend([nn.Dropout(dropout), nn.Linear(router_hidden_dim, router_hidden_dim, dtype=dtype), nn.ReLU()])
            layers.extend([nn.Dropout(dropout), nn.Linear(router_hidden_dim, num_experts, dtype=dtype)])
            self.mlp = nn.Sequential(*layers)

        self.task_weight: Optional[torch.Tensor] = None

    def forward(self, task_presentation: torch.Tensor) -> torch.Tensor:
        self.task_weight = F.softmax(self.mlp(task_presentation), dim=-1)
        return self.task_weight

    def get_task_weight(self) -> Optional[torch.Tensor]:
        return self.task_weight

    def divergence_loss(self) -> torch.Tensor:
        if self.task_weight is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        task_weight_batched = self.task_weight
        max_entropy = torch.log(torch.tensor(self.num_experts, dtype=task_weight_batched.dtype, device=task_weight_batched.device))
        max_entropy_m = self.gamma_div_balance * max_entropy
        min_entropy_p = self.gamma_div_certain * max_entropy
        max_div = max_entropy_m - min_entropy_p

        mean_gate = torch.mean(task_weight_batched, dim=0)
        entropy_m = -torch.sum(mean_gate * torch.log(mean_gate + 1e-9), dim=-1)
        entropy_m = torch.clamp(entropy_m, max=max_entropy_m)

        entropy_p = -torch.sum(task_weight_batched * torch.log(task_weight_batched + 1e-9), dim=-1)
        entropy_p = torch.clamp(entropy_p, min=min_entropy_p)
        entropy_p = torch.mean(entropy_p, dim=-1)

        return torch.relu(max_div - (entropy_m - entropy_p)) / max_entropy

    def clear(self) -> None:
        self.task_weight = None


class TokenRouter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        dropout: float,
        num_router_mlp_layers: int,
        router_hidden_dim: int,
        layer_id: int,
        max_layer_id: int,
        use_task_router: bool,
        task_router_only: bool,
        epsilon_alpha: float,
        alpha_shift: float,
        alpha_low_bound: float,
        alpha_up_bound: float,
        top_k_routing_strategy: bool,
        top_k: int,
        gamma_div_balance_t: float,
        gamma_div_certain_t: float,
        gamma_div_balance_s: float,
        gamma_div_certain_s: float,
        dtype: torch.dtype,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k_routing_strategy = top_k_routing_strategy
        self.top_k = top_k
        self.layer_id = layer_id
        self.gamma_div_balance = gamma_div_balance_t
        self.gamma_div_certain = gamma_div_certain_t

        safe_max_layer = max(max_layer_id, 1)
        alpha = -epsilon_alpha + 2 * epsilon_alpha * (layer_id / safe_max_layer) + alpha_shift
        alpha_tensor = torch.tensor(alpha, dtype=dtype)

        if use_task_router:
            if task_router_only:
                self.task_router = TaskRouter(
                    hidden_size,
                    num_experts,
                    dropout,
                    num_router_mlp_layers,
                    router_hidden_dim,
                    gamma_div_balance_s,
                    gamma_div_certain_s,
                    dtype,
                )
                self.task_router_only = True
                self.alpha = None
            else:
                alpha_ratio = torch.sigmoid(alpha_tensor)
                if alpha_ratio < alpha_low_bound:
                    self.task_router = None
                    self.task_router_only = False
                    self.alpha = None
                elif alpha_ratio > alpha_up_bound:
                    self.task_router = TaskRouter(
                        hidden_size,
                        num_experts,
                        dropout,
                        num_router_mlp_layers,
                        router_hidden_dim,
                        gamma_div_balance_s,
                        gamma_div_certain_s,
                        dtype,
                    )
                    self.task_router_only = True
                    self.alpha = None
                else:
                    self.task_router = TaskRouter(
                        hidden_size,
                        num_experts,
                        dropout,
                        num_router_mlp_layers,
                        router_hidden_dim,
                        gamma_div_balance_s,
                        gamma_div_certain_s,
                        dtype,
                    )
                    self.task_router_only = False
                    self.alpha = nn.Parameter(alpha_tensor)
        else:
            self.task_router = None
            self.task_router_only = False
            self.alpha = None

        if not self.task_router_only:
            if num_router_mlp_layers == 1:
                self.mlp = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, num_experts, dtype=dtype),
                )
            else:
                layers: list[nn.Module] = [
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, router_hidden_dim, dtype=dtype),
                    nn.ReLU(),
                ]
                for _ in range(num_router_mlp_layers - 2):
                    layers.extend([nn.Dropout(dropout), nn.Linear(router_hidden_dim, router_hidden_dim, dtype=dtype), nn.ReLU()])
                layers.extend([nn.Dropout(dropout), nn.Linear(router_hidden_dim, num_experts, dtype=dtype)])
                self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = None

        self.routing_weight: Optional[torch.Tensor] = None
        self.token_routing_weight: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.task_router_only:
            task_weight = self.task_router.get_task_weight() if self.task_router is not None else None
            if task_weight is None:
                task_weight = torch.zeros(hidden_states.shape[0], self.num_experts, device=hidden_states.device, dtype=hidden_states.dtype)
                task_weight[:, 0] = 1.0
            routing_weight = task_weight.unsqueeze(-2).expand(hidden_states.shape[:-1] + (self.num_experts,))
            self.routing_weight = routing_weight
        else:
            token_weight = F.softmax(self.mlp(hidden_states), dim=-1)
            self.token_routing_weight = token_weight
            if self.task_router is not None:
                task_weight = self.task_router.get_task_weight()
                if task_weight is None:
                    task_weight = torch.zeros(token_weight.shape[0], self.num_experts, device=token_weight.device, dtype=token_weight.dtype)
                    task_weight[:, 0] = 1.0
                alpha = torch.sigmoid(self.alpha)
                self.routing_weight = (1 - alpha) * token_weight + alpha * task_weight.unsqueeze(-2)
            else:
                self.routing_weight = token_weight

        if self.top_k_routing_strategy and self.top_k > 0:
            top_k = min(self.top_k, self.num_experts)
            top_k_values, top_k_indices = torch.topk(self.routing_weight, top_k, dim=-1)
            sparse = torch.full_like(self.routing_weight, torch.finfo(self.routing_weight.dtype).min)
            sparse.scatter_(-1, top_k_indices, top_k_values)
            self.routing_weight = torch.softmax(sparse, dim=-1)

        return self.routing_weight

    def get_routing_weight(self) -> Optional[torch.Tensor]:
        return self.routing_weight

    def divergence_loss(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.task_router_only:
            if self.routing_weight is None:
                return torch.tensor(0.0, device=attention_mask.device)
            return torch.tensor(0.0, dtype=self.routing_weight.dtype, device=self.routing_weight.device)

        if self.token_routing_weight is None:
            return torch.tensor(0.0, device=attention_mask.device)

        token_routing_weight = self.token_routing_weight
        mask = attention_mask.to(token_routing_weight.dtype).unsqueeze(-1)
        max_entropy = torch.log(torch.tensor(self.num_experts, dtype=token_routing_weight.dtype, device=token_routing_weight.device))
        max_entropy_m = self.gamma_div_balance * max_entropy
        min_entropy_p = self.gamma_div_certain * max_entropy
        max_div = max_entropy_m - min_entropy_p

        num_token = torch.sum(mask).clamp_min(1.0)
        token_routing_weight = token_routing_weight * mask
        mean_gate = torch.sum(token_routing_weight.view(-1, self.num_experts), dim=0) / num_token

        entropy_m = -torch.sum(mean_gate * torch.log(mean_gate + 1e-9), dim=-1)
        entropy_m = torch.clamp(entropy_m, max=max_entropy_m)

        entropy_p = -torch.sum(token_routing_weight * torch.log(token_routing_weight + 1e-9), dim=-1)
        entropy_p = torch.clamp(entropy_p, min=min_entropy_p) * mask.squeeze(-1)
        entropy_p = torch.sum(entropy_p) / num_token

        return torch.relu(max_div - (entropy_m - entropy_p)) / max_entropy

    def load_balancing_loss(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.token_routing_weight is None or self.routing_weight is None:
            return torch.tensor(0.0, device=attention_mask.device)

        routing_weight = self.token_routing_weight
        mask = attention_mask.to(routing_weight.dtype)
        num_token = mask.sum().clamp_min(1.0)
        routing_weight = routing_weight * mask.unsqueeze(-1)

        count = torch.sign(self.routing_weight * mask.unsqueeze(-1))
        divisor = max(float(self.top_k), 1.0)
        freq = torch.sum(count.view(-1, self.num_experts), dim=0) / (num_token * divisor)
        prop = torch.sum(routing_weight.view(-1, self.num_experts), dim=0) / num_token
        loss = torch.sum(prop * freq) * self.num_experts
        return loss.unsqueeze(0)

    def clear(self) -> None:
        if self.task_router is not None:
            self.task_router.clear()
        self.routing_weight = None
        self.token_routing_weight = None


class HMoRaLayer(LoraLayer, ABC):
    adapter_layer_names = LoraLayer.adapter_layer_names
    other_param_names = LoraLayer.other_param_names + (
        "num_experts",
        "use_hydra_lora",
        "use_single_lora",
        "router_key",
        "router_use_cache",
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__(base_layer, **kwargs)
        self.num_experts = {}
        self.use_hydra_lora = {}
        self.use_single_lora = {}
        self.router_key = {}
        self.router_use_cache = {}
        self._hmora_parent_ref = None

    def set_hmora_parent(self, parent_model: nn.Module) -> None:
        self._hmora_parent_ref = weakref.ref(parent_model)

    def _get_parent(self):
        return None if self._hmora_parent_ref is None else self._hmora_parent_ref()

    def update_layer(
        self,
        adapter_name: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: bool,
        num_experts: int,
        use_hydra_lora: bool,
        use_single_lora: bool,
        router_key: Optional[str],
        router_use_cache: bool,
    ) -> None:
        if lora_rank <= 0:
            raise ValueError(f"The rank `r` should be a positive integer value but got {lora_rank}.")
        if num_experts <= 0:
            raise ValueError(f"`num_experts` must be positive, got {num_experts}.")

        self.r[adapter_name] = lora_rank
        self.lora_alpha[adapter_name] = lora_alpha
        self.num_experts[adapter_name] = num_experts
        self.use_hydra_lora[adapter_name] = use_hydra_lora
        self.use_single_lora[adapter_name] = use_single_lora
        self.router_key[adapter_name] = router_key
        self.router_use_cache[adapter_name] = router_use_cache

        if lora_dropout > 0.0:
            self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout[adapter_name] = nn.Identity()

        if use_single_lora:
            self.lora_A[adapter_name] = nn.Linear(self.in_features, lora_rank, bias=False)
            self.lora_B[adapter_name] = nn.Linear(lora_rank, self.out_features, bias=False)
        else:
            in_rank = lora_rank if use_hydra_lora else lora_rank * num_experts
            self.lora_A[adapter_name] = nn.Linear(self.in_features, in_rank, bias=False)
            self.lora_B[adapter_name] = nn.Linear(lora_rank * num_experts, self.out_features, bias=False)

        self.scaling[adapter_name] = lora_alpha / math.sqrt(lora_rank)

        self.reset_lora_parameters(adapter_name, init_lora_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights) -> None:
        if init_lora_weights is False:
            return
        if adapter_name not in self.lora_A or adapter_name not in self.lora_B:
            return

        if init_lora_weights is True:
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "gaussian":
            nn.init.normal_(self.lora_A[adapter_name].weight, std=1.0 / max(1, self.r[adapter_name]))
        else:
            raise ValueError(f"Unsupported HMoRA initialization: {init_lora_weights}")
        nn.init.zeros_(self.lora_B[adapter_name].weight)

    def _get_router(self, adapter_name: str) -> Optional[TokenRouter]:
        parent = self._get_parent()
        if parent is None:
            return None
        key = self.router_key.get(adapter_name)
        if key is None:
            return None
        return parent.get_router(adapter_name, key)


class LinearHMoRaLayer(nn.Module, HMoRaLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        lora_rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        num_experts: int = 8,
        use_hydra_lora: bool = False,
        use_single_lora: bool = False,
        router_key: Optional[str] = None,
        router_use_cache: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        HMoRaLayer.__init__(self, base_layer=base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            num_experts=num_experts,
            use_hydra_lora=use_hydra_lora,
            use_single_lora=use_single_lora,
            router_key=router_key,
            router_use_cache=router_use_cache,
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        pass

    def unmerge(self) -> None:
        pass

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)

        if self.disable_adapters:
            return result.to(previous_dtype)

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue

            x_cast = x.to(self.lora_A[active_adapter].weight.dtype)
            dropped = self.lora_dropout[active_adapter](x_cast)

            if self.use_single_lora[active_adapter]:
                delta = self.lora_B[active_adapter](self.lora_A[active_adapter](dropped)) * self.scaling[active_adapter]
                result = result + delta.to(result.dtype)
                continue

            router = self._get_router(active_adapter)
            if router is None:
                gate_shape = x_cast.shape[:-1] + (self.num_experts[active_adapter],)
                gate = torch.zeros(gate_shape, device=x_cast.device, dtype=x_cast.dtype)
                gate[..., 0] = 1.0
            else:
                gate = None
                if self.router_use_cache[active_adapter]:
                    gate = router.get_routing_weight()
                    if gate is not None and gate.shape[:-1] != x_cast.shape[:-1]:
                        gate = None
                if gate is None:
                    gate = router(x_cast)
                gate = gate.to(x_cast.dtype)

            hidden_states = F.linear(dropped, self.lora_A[active_adapter].weight)
            rank = self.r[active_adapter]
            num_experts = self.num_experts[active_adapter]
            target_shape = hidden_states.shape[:-1] + (num_experts, rank)
            if self.use_hydra_lora[active_adapter]:
                hidden_states = hidden_states.unsqueeze(-2).expand(target_shape)
            else:
                hidden_states = hidden_states.view(target_shape)

            hidden_states = (hidden_states * gate.unsqueeze(-1)).reshape(hidden_states.shape[:-2] + (-1,))
            delta = F.linear(hidden_states, self.lora_B[active_adapter].weight) * self.scaling[active_adapter]
            result = result + delta.to(result.dtype)

        return result.to(previous_dtype)
