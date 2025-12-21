# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from peft.metrics import record_moelpr_language_routing, record_moelpr_scalars
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers.utils import logging as hf_logging

logger = hf_logging.get_logger("peft.moelpr")


class MoeLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("moe_experts", "moe_router_embedding")
    other_param_names = ("num_experts")

    def __init__(self, base_layer: nn.Module) -> None:
        self.base_layer = base_layer
        self.num_experts = {}
        self.moe_router_embedding = nn.ModuleDict({})
        self.moe_experts = nn.ModuleDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        if hasattr(base_layer, "gate_proj"):
            self.in_features = base_layer.gate_proj.in_features
        elif hasattr(base_layer, "fc1"):
            self.in_features = base_layer.fc1.in_features
        else:
            raise NotImplementedError

    def update_layer(self, base_layer, adapter_name, num_experts, init_moe_weights):
        self.num_experts[adapter_name] = num_experts
        self.moe_router_embedding[adapter_name] = nn.Linear(self.in_features, num_experts, bias=False)
        self.moe_experts[adapter_name] = nn.ModuleList([copy.deepcopy(base_layer) for _ in range(num_experts - 1)])

        if init_moe_weights:
            self.reset_moe_parameters(adapter_name)

        if hasattr(base_layer, "gate_proj"):
            weight = base_layer.gate_proj.weight
        elif hasattr(base_layer, "fc1"):
            weight = base_layer.fc1.weight
        else:
            raise NotImplementedError
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def reset_moe_parameters(self, adapter_name):
        if adapter_name in self.moe_router_embedding.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.xavier_normal_(self.moe_router_embedding[adapter_name].weight)


class MLP(nn.Module, MoeLayer):
    # Moe implemented in a mlp layer
    def __init__(
            self,
            base_layer,
            adapter_name: str,
            num_experts: int = 2,
            init_moe_weights: bool = True,
            topk: int = None,
            aux_loss_coef: float = None,
            lpr_loss_coef: float = None,
            moelpr_debug_mode: bool = False,
            **kwargs,
    ) -> None:
        super().__init__()
        MoeLayer.__init__(self, base_layer)

        self.aux_loss_coef = aux_loss_coef
        self.topk = topk
        self.lpr_loss_coef = lpr_loss_coef
        self.debug_mode = moelpr_debug_mode
        self._active_adapter = adapter_name
        self.update_layer(base_layer, adapter_name, num_experts, init_moe_weights)
        self.latest_router_logits = None

    @staticmethod
    def _flatten_batch_feature(value: Optional[torch.Tensor], batch_size: int, sequence_length: int) -> Optional[torch.Tensor]:
        if value is None or not torch.is_tensor(value):
            return None
        tensor = value
        if tensor.dim() == 0:
            tensor = tensor.view(1, 1).expand(batch_size, sequence_length)
        elif tensor.dim() == 1:
            if tensor.size(0) == batch_size * sequence_length:
                return tensor
            if tensor.size(0) == batch_size:
                tensor = tensor[:, None].expand(-1, sequence_length)
            else:
                return None
        elif tensor.dim() == 2 and tensor.size(1) != sequence_length:
            tensor = tensor[:, :sequence_length]
        return tensor.reshape(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype
        adapter = self.active_adapters[0]
        router = self.moe_router_embedding[adapter]  # b x s x e
        result, router_logits = self.topk_route(x, router, adapter)
        self.latest_router_logits = router_logits
        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "moe." + rep

    def topk_route(self, hidden_states, router, adapter=None):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = router(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.topk, dim=-1)
        if self.topk != 1:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        if self.debug_mode:
            num_experts = router_logits.shape[1]
            expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
            logger.info("[MOELPR DEBUG] Expert counts: %s", expert_counts.tolist())

            if hasattr(self, "lang_mask") and self.lang_mask is not None:
                lang_mask_flat = self.lang_mask.view(-1)
                if lang_mask_flat.shape[0] == selected_experts.shape[0]:
                    original_lang_tokens = selected_experts[lang_mask_flat == 1]
                    new_lang_tokens = selected_experts[lang_mask_flat == 0]

                    if original_lang_tokens.numel() > 0:
                        original_counts = torch.bincount(original_lang_tokens.reshape(-1), minlength=num_experts)
                        logger.info("[MOELPR DEBUG] Original language expert counts: %s", original_counts.tolist())
                    if new_lang_tokens.numel() > 0:
                        new_counts = torch.bincount(new_lang_tokens.reshape(-1), minlength=num_experts)
                        logger.info("[MOELPR DEBUG] New language expert counts: %s", new_counts.tolist())

        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts[adapter]).permute(2, 1,
                                                                                                                   0)

        experts = [self.base_layer] + [k for k in self.moe_experts[adapter]]
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts[adapter]):
            expert_layer = experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        with torch.no_grad():
            token_count = routing_weights.size(0)
            if token_count > 0:
                expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=self.num_experts[adapter])
                load_cv = 0.0
                if expert_counts.numel() > 0:
                    mean_load = expert_counts.float().mean()
                    if mean_load > 0:
                        load_cv = float(
                            (expert_counts.float().std(unbiased=False) / (mean_load + 1e-6)).item()
                        )
                entropy = (-routing_weights.float() * torch.log(routing_weights.float() + 1e-8)).sum(dim=-1).mean()
                record_moelpr_scalars(
                    {
                        "expert_load_cv": load_cv,
                        "router_entropy": float(entropy.item()),
                    },
                    weight=float(token_count),
                )
                language_ids = self._flatten_batch_feature(getattr(self, "language_ids", None), batch_size, sequence_length)
                lang_mask = self._flatten_batch_feature(getattr(self, "lang_mask", None), batch_size, sequence_length)
                attention = self._flatten_batch_feature(getattr(self, "attention_mask", None), batch_size, sequence_length)
                record_moelpr_language_routing(
                    language_ids=language_ids,
                    selected_experts=selected_experts,
                    routing_weights=routing_weights,
                    num_experts=self.num_experts[adapter],
                    lang_mask=lang_mask,
                    attention_mask=attention,
                )
        return final_hidden_states, router_logits

""" NOTE: COPIED FROM ORIGINAL IMPLEMENTATION """
def lpr_loss_func(router_weights, lang_mask):
    # router_weights (n x 2) x 32
    loss_func = CrossEntropyLoss()
    router_weights = torch.stack(router_weights)  # 32 x n x e
    router_weights = torch.nn.functional.softmax(router_weights, dim=-1, dtype=torch.float)
    mask = lang_mask.reshape(-1).bool().expand(router_weights.size()[:2])
    probs = router_weights[mask].to(torch.float).contiguous()  # n x e
    target = torch.zeros_like(probs, dtype=torch.long)[:, 0].contiguous()
    loss = loss_func(probs, target)
    return loss


def load_balancing_loss_func(
        gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2,
        attention_mask: Optional[torch.Tensor] = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1, dtype=torch.float)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_s, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_s * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_s, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_s, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts
