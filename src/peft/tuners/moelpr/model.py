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
import operator
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import reduce
from itertools import chain
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_auto_gptq_quant_linear,
    get_quantization_config,
)
from transformers.utils import logging as hf_logging

logger = hf_logging.get_logger("peft.moelpr")
from .config import MoeConfig
from .layer import MLP, MoeLayer, lpr_loss_func, load_balancing_loss_func


class MoeModel(BaseTuner):
    prefix: str = "moe_"

    def __init__(self, model, config, adapter_name) -> None:
        moe_layers = config.layers_to_transform if isinstance(config, MoeConfig) else config[adapter_name].layers_to_transform
        self.moe_layers = moe_layers.copy()
        self.moe_mlp_layers = []
        #要进入tuner去替换
        self._cached_language_ids = None
        super().__init__(model, config, adapter_name)

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        language_ids = kwargs.get("language_ids")
        if language_ids is not None:
            self._cached_language_ids = language_ids
        try:
            yield
        finally:
            self._cached_language_ids = None

    def forward(self, *args, **kwargs):
        """
        This model is a wrapper around the base model.
        We need to forward all calls to the base model, but first we need to pass the lang_mask and attention_mask
        to the MLP layers so they can be accessed by the get_aux_loss method.
        """
        lang_mask = kwargs.get("lang_mask", None)
        attention_mask = kwargs.get("attention_mask", None)
        language_ids = kwargs.pop("language_ids", None)
        if language_ids is None:
            language_ids = self._cached_language_ids

        # Set masks on MLP layers
        for module in self.model.modules():
            if isinstance(module, MLP):
                module.lang_mask = lang_mask
                module.attention_mask = attention_mask
                module.language_ids = language_ids

        return self.model(*args, **kwargs)

    def get_aux_loss(self):
        """
        Calculates and returns the auxiliary loss for MoE-LPR based on the current training stage.
        """
        total_aux_loss = torch.tensor(0.0, device=self.device)
        active_adapter = self.active_adapter
        if isinstance(active_adapter, list):
            active_adapter = active_adapter[0]
        active_adapter_config = self.peft_config[active_adapter]
        stage = active_adapter_config.stage

        # No aux loss if not training or no coef is set
        if not self.training:
            return total_aux_loss

        all_router_logits = []
        for module in self.model.modules():
            if isinstance(module, MLP) and module.latest_router_logits is not None:
                all_router_logits.append(module.latest_router_logits)

        if not all_router_logits:
            return total_aux_loss

        if stage == 1 and active_adapter_config.aux_loss_coef is not None:
            # Stage 1: Load-balancing loss
            balance_loss = []
            for layer_router_logits in all_router_logits:
                # get attention_mask from the layer
                attention_mask = None
                for module in self.model.modules():
                    if isinstance(module, MLP):
                        attention_mask = module.attention_mask
                        break

                balance_loss.append(load_balancing_loss_func(
                    (layer_router_logits,),
                    active_adapter_config.num_experts,
                    active_adapter_config.topk,
                    attention_mask,
                ))
            balance_loss = torch.stack(balance_loss)
            balance_loss = torch.mean(balance_loss)
            total_aux_loss += active_adapter_config.aux_loss_coef * balance_loss

        elif stage == 2 and active_adapter_config.lpr_loss_coef is not None:
            # Stage 2: LPR loss
            lang_mask = None
            for module in self.model.modules():
                if isinstance(module, MLP):
                    lang_mask = module.lang_mask
                    break

            if lang_mask is None:
                warnings.warn("LPR loss requires a `lang_mask` but it was not provided. Skipping LPR loss.")
                return total_aux_loss

            lpr_loss = lpr_loss_func(
                all_router_logits,
                lang_mask=lang_mask,
            )
            total_aux_loss += active_adapter_config.lpr_loss_coef * lpr_loss
            if active_adapter_config.moelpr_debug_mode:
                logger.info("[MOELPR DEBUG] Stage 2 LPR loss: %.4f", total_aux_loss.item())

        return total_aux_loss

    def _check_new_adapter_config(self, config: MoeConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )
    #检查哪些模块是我们需要替换的
    def _check_target_module_exists(self, moe_config, key):
        if "mlp" in key.split("."):
            layerid = int(key.split(".")[2])
            if layerid in self.moe_layers:
                self.moe_layers.pop(self.moe_layers.index(layerid))
                return True
        return False

    def _create_and_replace(
        self,
        moe_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        layerid=None,
        **optional_kwargs,
    ):
        # parent --> LlamaMLP
        # decoderlayer_module --> LlamaDecoderLayer
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        layerid = current_key.split(".")[2]
        # TODO: better deal with that
        #decoderlayer_module = self.model.get_submodule(".".join(current_key.split(".")[:-2]))
        #mlp_name = current_key.split(".")[-2]
        new_module = self._create_new_module(moe_config, adapter_name, target, layerid=layerid)
        self.moe_mlp_layers.append(new_module)
        if adapter_name != self.active_adapter:
            # adding an additional adapter: it is not automatically trainable
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name):
                if hasattr(child, "gate_proj"):
                    weight = child.gate_proj.weight
                elif hasattr(child, "fc1"):
                    weight = child.fc1.weight
                else:
                    raise NotImplementedError
                module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model) -> None:
        r"""
        Stage-aware freezing: during Stage 1 we train all inserted experts + router; in Stage 2 only the router
        embeddings remain trainable so we can \"review\" routing without drifting expert weights.
        """
        active_adapter = self.active_adapter
        if isinstance(active_adapter, list):
            active_adapter = active_adapter[0]
        stage = getattr(self.peft_config[active_adapter], "stage", 1)

        for name, param in model.named_parameters():
            if self.prefix not in name:
                param.requires_grad = False
                continue

            if stage == 2:
                # keep router embeddings trainable, freeze newly added experts to mimic the published workflow
                if ".moe_router_embedding" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

    @staticmethod
    def _create_new_module(moe_config, adapter_name, target, **kwargs):
        new_module = MLP(target, adapter_name,
                         num_experts=moe_config.num_experts,
                         init_moe_weights=moe_config.init_moe_weights,
                         topk=moe_config.topk,
                         aux_loss_coef=moe_config.aux_loss_coef,
                         lpr_loss_coef=moe_config.lpr_loss_coef,
                         moelpr_debug_mode=moe_config.moelpr_debug_mode,
                         **kwargs)
        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, MoeLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        return peft_config
