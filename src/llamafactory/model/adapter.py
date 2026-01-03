# Copyright 2025 the LlamaFactory team.
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

import os
import re
from distutils.util import strtobool
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
from peft import (
    AdaMoleConfig,
    ColaConfig,
    HydraLoraConfig,
    LoraConfig,
    LoraModel,
    MoelprConfig,
    MolaConfig,
    OFTConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from transformers.integrations import is_deepspeed_zero3_enabled

from ..extras import logging
from ..extras.constants import EngineName
from ..extras.language import load_language_groupings, load_language_map
from .model_utils.ktransformers import get_kt_peft_model, load_kt_peft_model
from .model_utils.misc import find_all_linear_modules, find_expanded_modules
from .model_utils.quantization import QuantizationMethod
from .model_utils.unsloth import get_unsloth_peft_model, load_unsloth_peft_model
from .model_utils.visual import COMPOSITE_MODELS, get_forbidden_modules, patch_target_modules


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _accelerate_wants_fsdp() -> bool:
    return strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1


def _build_language_metadata(language_map_spec: Optional[str]):
    language_map, families, subgroup_sizes, language_to_subgroup = load_language_groupings(language_map_spec)
    if not language_map:
        return None, None, None, None, None

    languages = sorted(language_map.keys())
    if families is None:
        families = sorted(set(language_map.values()))
    family_to_idx = {family: idx for idx, family in enumerate(families)}
    language_to_family_ids = [family_to_idx[language_map[lang]] for lang in languages]

    if language_to_subgroup is None:
        language_to_subgroup_ids = None
    else:
        language_to_subgroup_ids = [language_to_subgroup.get(lang, -1) for lang in languages]

    return languages, families, language_to_family_ids, subgroup_sizes, language_to_subgroup_ids


def _parse_optional_int_list(value: Optional[Union[str, Sequence[int]]]) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = list(value)
    if not items:
        return None
    parsed = []
    for item in items:
        parsed.append(int(item))
    return parsed


def _setup_full_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Full")
    forbidden_modules = get_forbidden_modules(model.config, finetuning_args)
    for name, param in model.named_parameters():
        if not any(forbidden_module in name for forbidden_module in forbidden_modules):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)


def _setup_freeze_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Freeze")
    if hasattr(model.config, "text_config"):  # composite models
        config = getattr(model.config, "text_config")
    else:
        config = model.config

    num_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
        or getattr(config, "n_layer", None)
    )
    if not num_layers:
        raise ValueError("Current model does not support freeze tuning.")

    if finetuning_args.use_llama_pro:
        if num_layers % finetuning_args.freeze_trainable_layers != 0:
            raise ValueError(
                f"`num_layers` {num_layers} should be "
                f"divisible by `num_layer_trainable` {finetuning_args.freeze_trainable_layers}."
            )

        stride = num_layers // finetuning_args.freeze_trainable_layers
        trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)
    elif finetuning_args.freeze_trainable_layers > 0:  # fine-tuning the last n layers if num_layer_trainable > 0
        trainable_layer_ids = range(max(0, num_layers - finetuning_args.freeze_trainable_layers), num_layers)
    else:  # fine-tuning the first n layers if num_layer_trainable < 0
        trainable_layer_ids = range(min(-finetuning_args.freeze_trainable_layers, num_layers))

    hidden_modules = set()
    non_hidden_modules = set()
    for name, _ in model.named_parameters():
        if ".0." in name:
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])
        elif ".1." in name:  # MoD starts from layer 1
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])

        if re.search(r"\.\d+\.", name) is None:
            non_hidden_modules.add(name.split(".")[-2])  # remove weight/bias

    trainable_layers = []
    for module_name in finetuning_args.freeze_trainable_modules:
        if module_name != "all" and module_name not in hidden_modules:
            raise ValueError(
                "Module {} is not found, please choose from {}".format(module_name, ", ".join(hidden_modules))
            )

        for idx in trainable_layer_ids:
            trainable_layers.append(".{:d}.{}".format(idx, module_name if module_name != "all" else ""))

    if finetuning_args.freeze_extra_modules:
        for module_name in finetuning_args.freeze_extra_modules:
            if module_name not in non_hidden_modules:
                raise ValueError(
                    "Module {} is not found, please choose from {}".format(module_name, ", ".join(non_hidden_modules))
                )

            trainable_layers.append(module_name)

    model_type = getattr(model.config, "model_type", None)
    if not finetuning_args.freeze_multi_modal_projector and model_type in COMPOSITE_MODELS:
        trainable_layers.append(COMPOSITE_MODELS[model_type].projector_key)

    forbidden_modules = get_forbidden_modules(model.config, finetuning_args)
    for name, param in model.named_parameters():
        if any(trainable_layer in name for trainable_layer in trainable_layers) and not any(
            forbidden_module in name for forbidden_module in forbidden_modules
        ):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)

    logger.info_rank0("Set trainable layers: {}".format(",".join(trainable_layers)))


def _setup_lora_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    if is_trainable:
        if finetuning_args.finetuning_type == "oft":
            logger.info_rank0("Fine-tuning method: OFT")
        else:
            logger.info_rank0("Fine-tuning method: {}".format("DoRA" if finetuning_args.use_dora else "LoRA"))

    adapter_to_resume = None
    language_map = None
    target_modules: list[str] = []

    if model_args.adapter_name_or_path is not None:
        is_mergeable = True
        if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False

        if model_args.use_kt:
            assert len(model_args.adapter_name_or_path) == 1, "KTransformers model only accepts a single adapter"
            is_mergeable = False

        if model_args.use_unsloth:
            assert len(model_args.adapter_name_or_path) == 1, "Unsloth model only accepts a single adapter."
            is_mergeable = False

        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
        else:
            adapter_to_merge = model_args.adapter_name_or_path

        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "offload_folder": model_args.offload_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }

        if model_args.use_kt:
            if model_args.infer_backend != EngineName.KT:
                raise ValueError(
                    "We should use ktransformers as backend to infer the adapter fine-tuned by ktransformers."
                )

        for adapter in adapter_to_merge:
            model: LoraModel = PeftModel.from_pretrained(model, adapter, **init_kwargs)
            model = model.merge_and_unload()

        if len(adapter_to_merge) > 0:
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")

        if adapter_to_resume is not None:  # resume lora training
            if model_args.use_kt:
                model = load_kt_peft_model(model_args, model)
            elif model_args.use_unsloth:
                model = load_unsloth_peft_model(config, model_args, finetuning_args, is_trainable=is_trainable)
            else:
                model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable, **init_kwargs)

        logger.info_rank0("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    if is_trainable and adapter_to_resume is None:  # create new lora weights while training
        if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
        else:
            target_modules = finetuning_args.lora_target

        if model_args.use_kt:
            new_list = []
            for m in target_modules:
                if m in ("down_proj", "up_proj", "gate_proj"):
                    new_list.extend([f"mlp.{m}", f"shared_experts.{m}"])
                elif m not in ("generate_linear", "orig_module", "prefill_linear"):
                    new_list.append(m)

            target_modules[:] = new_list

        if finetuning_args.use_llama_pro:
            target_modules = find_expanded_modules(model, target_modules, finetuning_args.freeze_trainable_layers)

        target_modules = patch_target_modules(model, finetuning_args, target_modules)

        if (
            finetuning_args.use_dora
            and getattr(model, "quantization_method", None) is not None
            and getattr(model, "quantization_method", None) != QuantizationMethod.BNB
        ):
            raise ValueError("DoRA is not compatible with PTQ-quantized models.")

        if model_args.resize_vocab and finetuning_args.additional_target is None:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            module_names = set()
            for name, module in model.named_modules():
                if module in [input_embeddings, output_embeddings]:
                    module_names.add(name.split(".")[-1])

            finetuning_args.additional_target = module_names
            logger.warning_rank0("Vocab has been resized, add {} to trainable params.".format(",".join(module_names)))

        if finetuning_args.finetuning_type == "lora":
            peft_kwargs = {
                "r": finetuning_args.lora_rank,
                "target_modules": target_modules,
                "lora_alpha": finetuning_args.lora_alpha,
                "lora_dropout": finetuning_args.lora_dropout,
                "use_rslora": finetuning_args.use_rslora,
                "use_dora": finetuning_args.use_dora,
                "modules_to_save": finetuning_args.additional_target,
            }
        elif finetuning_args.finetuning_type == "oft":
            peft_kwargs = {
                "r": finetuning_args.oft_rank,
                "oft_block_size": finetuning_args.oft_block_size,
                "target_modules": target_modules,
                "module_dropout": finetuning_args.module_dropout,
                "modules_to_save": finetuning_args.additional_target,
            }

        if model_args.use_kt:
            if finetuning_args.finetuning_type == "oft":
                raise ValueError("KTransformers is currently not supported for OFT.")
            if finetuning_args.finetuning_type == "lora":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **peft_kwargs,
                )
            else:
                raise ValueError("KTransformers is currently only supported for LoRA.")

            model = get_kt_peft_model(model, peft_config)
            print(f"KT_model:{model}")
        elif model_args.use_unsloth:
            if finetuning_args.finetuning_type == "oft":
                raise ValueError("Unsloth is currently not supported for OFT.")

            model = get_unsloth_peft_model(model, model_args, peft_kwargs)
        else:
            if finetuning_args.pissa_init:
                if finetuning_args.pissa_iter == -1:
                    logger.info_rank0("Using PiSSA initialization.")
                    peft_kwargs["init_lora_weights"] = "pissa"
                else:
                    logger.info_rank0(f"Using PiSSA initialization with FSVD steps {finetuning_args.pissa_iter}.")
                    peft_kwargs["init_lora_weights"] = f"pissa_niter_{finetuning_args.pissa_iter}"

            if finetuning_args.finetuning_type == "lora":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **peft_kwargs,
                )
            elif finetuning_args.finetuning_type == "oft":
                peft_config = OFTConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **peft_kwargs,
                )
            model = get_peft_model(model, peft_config)

    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    if (
        is_trainable
        and (is_deepspeed_zero3_enabled() or _accelerate_wants_fsdp())
        and getattr(model_args, "compute_dtype", None) in (torch.float16, torch.bfloat16)
    ):
        target_dtype = model_args.compute_dtype
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            if param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)
        logger.info_rank0(f"FSDP/ZeRO-3 detected, casting trainable params to {target_dtype}.")

    return model


def _setup_cola_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    if model_args.use_kt or model_args.use_unsloth:
        raise ValueError("CoLA is not compatible with KTransformers or Unsloth.")

    if is_trainable:
        logger.info_rank0("Fine-tuning method: COLA")

    cola_debug = getattr(finetuning_args, "cola_debug", False)
    language_map = None
    target_modules: list[str] = []
    expected_experts: Optional[int] = None
    expected_heads: Optional[list[int]] = None
    if cola_debug:
        logger.info_rank0("[COLA DEBUG] Enabled CoLA architecture + expert init verification.")

    adapter_to_resume = None

    if model_args.adapter_name_or_path is not None:
        is_mergeable = True
        if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False

        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
        else:
            adapter_to_merge = model_args.adapter_name_or_path

        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "offload_folder": model_args.offload_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }
        for adapter in adapter_to_merge:
            model: "LoraModel" = PeftModel.from_pretrained(model, adapter, **init_kwargs)
            model = model.merge_and_unload()

        if len(adapter_to_merge) > 0:
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")

        if adapter_to_resume is not None:  # resume adapter training
            model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable, **init_kwargs)

        logger.info_rank0("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    if is_trainable and adapter_to_resume is None:
        if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
        else:
            target_modules = finetuning_args.lora_target

        if finetuning_args.use_llama_pro:
            target_modules = find_expanded_modules(model, target_modules, finetuning_args.freeze_trainable_layers)

        target_modules = patch_target_modules(model, finetuning_args, target_modules)

        if model_args.resize_vocab and finetuning_args.additional_target is None:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            module_names = set()
            for name, module in model.named_modules():
                if module in [input_embeddings, output_embeddings]:
                    module_names.add(name.split(".")[-1])

            finetuning_args.additional_target = module_names
            logger.warning_rank0("Vocab has been resized, add {} to trainable params.".format(",".join(module_names)))

        expert_num_A = _parse_optional_int_list(finetuning_args.cola_expert_num_A)
        expert_num_B = _parse_optional_int_list(finetuning_args.cola_expert_num_B)

        peft_kwargs = {
            "r": finetuning_args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": finetuning_args.lora_alpha,
            "lora_dropout": finetuning_args.lora_dropout,
            "num_A": finetuning_args.num_A,
            "num_B": finetuning_args.num_B,
            "expert_num_A": expert_num_A,
            "expert_num_B": expert_num_B,
            "use_cola_experts": finetuning_args.use_cola_experts,
            "cola_num_experts": finetuning_args.cola_num_experts,
            "cola_top_k": finetuning_args.cola_top_k,
            "cola_debug": finetuning_args.cola_debug,
            "cola_strategy": finetuning_args.cola_strategy,
            "modules_to_save": finetuning_args.additional_target,
        }
        language_map = load_language_map(finetuning_args.language_map)
        language_list, family_list, language_to_family, subgroup_sizes, language_to_subgroup_ids = _build_language_metadata(
            finetuning_args.language_map
        )
        if family_list:
            expected_experts = len(family_list)
        if finetuning_args.use_cola_experts and expected_experts:
            if finetuning_args.cola_num_experts != expected_experts:
                raise ValueError(
                    "CoLA expert config mismatch: "
                    f"cola_num_experts={finetuning_args.cola_num_experts} "
                    f"but language_map defines {expected_experts} groups."
                )
        if expert_num_B is None and subgroup_sizes and any(size > 0 for size in subgroup_sizes):
            expert_num_B = subgroup_sizes
        if expert_num_B is None and finetuning_args.use_cola_experts and expected_experts:
            expert_num_B = [finetuning_args.num_B] * expected_experts
        if finetuning_args.use_cola_experts and expert_num_B is not None and expected_experts:
            if len(expert_num_B) != expected_experts:
                raise ValueError(
                    "CoLA expert head config mismatch: "
                    f"expected {expected_experts} entries but got {len(expert_num_B)}."
                )
            if any(count <= 0 for count in expert_num_B):
                raise ValueError("CoLA expert head config contains non-positive counts.")
            expected_heads = list(expert_num_B)
        peft_kwargs.update(
            {
                "language_map": language_map,
                "language_list": language_list,
                "family_list": family_list,
                "language_to_family_ids": language_to_family,
                "language_to_subgroup_ids": language_to_subgroup_ids,
                "language_column": finetuning_args.language_column,
                "language_router_mode": finetuning_args.language_router_mode,
                "language_head_router_mode": finetuning_args.language_head_router_mode,
                "language_guidance_scope": finetuning_args.language_guidance_scope,
                "language_prior_weight": finetuning_args.language_prior_weight,
                "language_bias_value": finetuning_args.language_bias_value,
                "language_head_bias_value": finetuning_args.language_head_bias_value,
            }
        )
        init_lora_weights = finetuning_args.cola_init_lora_weights
        if init_lora_weights is None:
            if finetuning_args.use_cola_pissa_init:
                init_lora_weights = (
                    "pissa" if finetuning_args.pissa_iter == -1 else f"pissa_niter_{finetuning_args.pissa_iter}"
                )
            else:
                init_lora_weights = True
        else:
            lowered = init_lora_weights.lower()
            if lowered in {"true", "default", "lora"}:
                init_lora_weights = True
            elif lowered in {"false", "none", "random"}:
                init_lora_weights = False
            elif lowered == "pissa" and finetuning_args.pissa_iter != -1:
                init_lora_weights = f"pissa_niter_{finetuning_args.pissa_iter}"
        peft_kwargs["init_lora_weights"] = init_lora_weights
        logger.info_rank0(f"CoLA adapter initialization method: {init_lora_weights}.")

        cola_config = ColaConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            **peft_kwargs,
        )
        model = get_peft_model(model, cola_config)

    if finetuning_args.use_cola_experts:
        sample_layer = next((m for _, m in model.named_modules() if hasattr(m, "use_cola_experts")), None)
        if sample_layer is not None:
            actual_experts = int(getattr(sample_layer, "num_experts", 0) or 0)
            actual_heads = []
            for idx in range(actual_experts):
                key = f"expert_{idx}"
                count = 0
                if hasattr(sample_layer, "num_B") and key in sample_layer.num_B:
                    count = int(sample_layer.num_B.get(key, 0) or 0)
                actual_heads.append(count)
            logger.info_rank0(
                "[COLA SETUP] experts=%s heads_per_expert=%s router_mode=%s head_router_mode=%s guidance=%s top_k=%s",
                actual_experts,
                actual_heads,
                finetuning_args.language_router_mode,
                finetuning_args.language_head_router_mode,
                finetuning_args.language_guidance_scope,
                finetuning_args.cola_top_k,
            )
            if expected_experts is not None and actual_experts != expected_experts:
                raise ValueError(
                    f"CoLA runtime experts mismatch: expected {expected_experts}, got {actual_experts}."
                )
            if expected_heads is not None and actual_heads != expected_heads:
                raise ValueError(
                    f"CoLA runtime head mismatch: expected {expected_heads}, got {actual_heads}."
                )

    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    if cola_debug:
        for _, module in model.named_modules():
            if hasattr(module, "use_cola_experts"):
                module.cola_debug = True
        logger.info_rank0("[COLA DEBUG] Attached cola_debug=True to all CoLA layers.")
        if language_map is not None:
            if isinstance(language_map, dict):
                language_count = len(language_map)
            elif isinstance(language_map, list):
                language_count = len(language_map)
            else:
                language_count = None
        else:
            language_count = None
        cola_layers = sum(1 for _, module in model.named_modules() if hasattr(module, "use_cola_experts"))
        logger.info_rank0(
            "[COLA DEBUG] Summary: layers=%d target_modules=%d languages=%s num_experts=%s top_k=%s num_A=%s num_B=%s",
            cola_layers,
            len(target_modules),
            "unknown" if language_count is None else str(language_count),
            finetuning_args.cola_num_experts,
            finetuning_args.cola_top_k,
            finetuning_args.num_A,
            finetuning_args.num_B,
        )

    return model


def _setup_hydralora_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    if model_args.use_kt or model_args.use_unsloth:
        raise ValueError("HydraLoRA is not compatible with KTransformers or Unsloth.")

    if is_trainable:
        logger.info_rank0("Fine-tuning method: HYDRALORA")

    adapter_to_resume = None
    expected_experts: Optional[int] = None
    expected_heads: Optional[list[int]] = None

    if model_args.adapter_name_or_path is not None:
        is_mergeable = True
        if getattr(model, "quantization_method", None):
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False

        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
        else:
            adapter_to_merge = model_args.adapter_name_or_path

        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "offload_folder": model_args.offload_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }
        for adapter in adapter_to_merge:
            model: "LoraModel" = PeftModel.from_pretrained(model, adapter, **init_kwargs)
            model = model.merge_and_unload()

        if len(adapter_to_merge) > 0:
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")

        if adapter_to_resume is not None:
            model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable, **init_kwargs)

        logger.info_rank0("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    if is_trainable and adapter_to_resume is None:
        if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
        else:
            target_modules = finetuning_args.lora_target

        if finetuning_args.use_llama_pro:
            target_modules = find_expanded_modules(model, target_modules, finetuning_args.freeze_trainable_layers)

        target_modules = patch_target_modules(model, finetuning_args, target_modules)

        if model_args.resize_vocab and finetuning_args.additional_target is None:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            module_names = set()
            for name, module in model.named_modules():
                if module in [input_embeddings, output_embeddings]:
                    module_names.add(name.split(".")[-1])

            finetuning_args.additional_target = module_names
            logger.warning_rank0("Vocab has been resized, add {} to trainable params.".format(",".join(module_names)))

        expert_lora_nums = _parse_optional_int_list(finetuning_args.hydralora_expert_lora_nums)

        peft_kwargs = {
            "r": finetuning_args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": finetuning_args.lora_alpha,
            "lora_dropout": finetuning_args.lora_dropout,
            "lora_num": finetuning_args.lora_num,
            "expert_lora_nums": expert_lora_nums,
            "use_hydralora_experts": finetuning_args.use_hydralora_experts,
            "num_experts": finetuning_args.hydralora_num_experts,
            "top_k": finetuning_args.hydralora_top_k,
            "head_top_k": finetuning_args.hydralora_head_top_k,
            "hydralora_debug": finetuning_args.hydralora_debug,
            "modules_to_save": finetuning_args.additional_target,
        }
        language_map = load_language_map(finetuning_args.language_map)
        language_list, family_list, language_to_family, subgroup_sizes, language_to_subgroup_ids = _build_language_metadata(
            finetuning_args.language_map
        )
        if family_list:
            expected_experts = len(family_list)
        if finetuning_args.use_hydralora_experts and expected_experts:
            if finetuning_args.hydralora_num_experts != expected_experts:
                raise ValueError(
                    "Hydra expert config mismatch: "
                    f"hydralora_num_experts={finetuning_args.hydralora_num_experts} "
                    f"but language_map defines {expected_experts} groups."
                )
        if expert_lora_nums is None and subgroup_sizes and any(size > 0 for size in subgroup_sizes):
            expert_lora_nums = subgroup_sizes
        if expert_lora_nums is None and finetuning_args.use_hydralora_experts and expected_experts:
            expert_lora_nums = [finetuning_args.lora_num] * expected_experts
        if finetuning_args.use_hydralora_experts and expert_lora_nums is not None and expected_experts:
            if len(expert_lora_nums) != expected_experts:
                raise ValueError(
                    "Hydra expert head config mismatch: "
                    f"expected {expected_experts} entries but got {len(expert_lora_nums)}."
                )
            if any(count <= 0 for count in expert_lora_nums):
                raise ValueError("Hydra expert head config contains non-positive counts.")
            expected_heads = list(expert_lora_nums)
        peft_kwargs.update(
            {
                "language_map": language_map,
                "language_list": language_list,
                "family_list": family_list,
                "language_to_family_ids": language_to_family,
                "language_to_subgroup_ids": language_to_subgroup_ids,
                "language_column": finetuning_args.language_column,
                "language_router_mode": finetuning_args.language_router_mode,
                "language_head_router_mode": finetuning_args.language_head_router_mode,
                "language_guidance_scope": finetuning_args.language_guidance_scope,
                "language_prior_weight": finetuning_args.language_prior_weight,
                "language_bias_value": finetuning_args.language_bias_value,
                "language_head_bias_value": finetuning_args.language_head_bias_value,
            }
        )

        hydra_config = HydraLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            **peft_kwargs,
        )
        model = get_peft_model(model, hydra_config)

    if finetuning_args.use_hydralora_experts:
        sample_layer = next((m for _, m in model.named_modules() if hasattr(m, "use_hydralora_experts")), None)
        if sample_layer is not None:
            actual_experts = int(getattr(sample_layer, "num_experts", 0) or 0)
            actual_heads = []
            for idx in range(actual_experts):
                key = f"expert_{idx}"
                count = 0
                if hasattr(sample_layer, "lora_num") and key in sample_layer.lora_num:
                    count = int(sample_layer.lora_num.get(key, 0) or 0)
                actual_heads.append(count)
            logger.info_rank0(
                "[HYDRA SETUP] experts=%s heads_per_expert=%s router_mode=%s head_router_mode=%s guidance=%s top_k=%s head_top_k=%s",
                actual_experts,
                actual_heads,
                finetuning_args.language_router_mode,
                finetuning_args.language_head_router_mode,
                finetuning_args.language_guidance_scope,
                finetuning_args.hydralora_top_k,
                finetuning_args.hydralora_head_top_k,
            )
            if expected_experts is not None and actual_experts != expected_experts:
                raise ValueError(
                    f"Hydra runtime experts mismatch: expected {expected_experts}, got {actual_experts}."
                )
            if expected_heads is not None and actual_heads != expected_heads:
                raise ValueError(
                    f"Hydra runtime head mismatch: expected {expected_heads}, got {actual_heads}."
                )

    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    if getattr(finetuning_args, "hydralora_debug", False):
        if language_map is not None:
            if isinstance(language_map, dict):
                language_count = len(language_map)
            elif isinstance(language_map, list):
                language_count = len(language_map)
            else:
                language_count = None
        else:
            language_count = None
        hydra_layers = sum(1 for _, module in model.named_modules() if hasattr(module, "use_hydralora_experts"))
        logger.info_rank0(
            "[HYDRA DEBUG] Summary: layers=%d target_modules=%d languages=%s num_experts=%s top_k=%s lora_num=%s",
            hydra_layers,
            len(target_modules),
            "unknown" if language_count is None else str(language_count),
            finetuning_args.hydralora_num_experts,
            finetuning_args.hydralora_top_k,
            finetuning_args.lora_num,
        )

    return model


def _setup_adamole_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    if finetuning_args.use_dora or finetuning_args.use_rslora:
        raise ValueError("AdaMoLE currently does not support DoRA or rank-stabilized LoRA weights.")
    if model_args.use_unsloth or model_args.use_kt:
        raise ValueError("AdaMoLE is not compatible with KTransformers or Unsloth.")

    if is_trainable:
        logger.info_rank0("Fine-tuning method: AdaMoLE")

    adapter_to_resume = None

    if model_args.adapter_name_or_path is not None:
        is_mergeable = True
        if getattr(model, "quantization_method", None):
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False

        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
        else:
            adapter_to_merge = model_args.adapter_name_or_path

        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "offload_folder": model_args.offload_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }
        for adapter in adapter_to_merge:
            model: "LoraModel" = PeftModel.from_pretrained(model, adapter, **init_kwargs)
            model = model.merge_and_unload()

        if len(adapter_to_merge) > 0:
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")

        if adapter_to_resume is not None:  # resume adapter training
            model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable, **init_kwargs)

        logger.info_rank0("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    if is_trainable and adapter_to_resume is None:
        if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
        else:
            target_modules = finetuning_args.lora_target

        if finetuning_args.use_llama_pro:
            target_modules = find_expanded_modules(model, target_modules, finetuning_args.freeze_trainable_layers)

        target_modules = patch_target_modules(model, finetuning_args, target_modules)

        if model_args.resize_vocab and finetuning_args.additional_target is None:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            module_names = set()
            for name, module in model.named_modules():
                if module in [input_embeddings, output_embeddings]:
                    module_names.add(name.split(".")[-1])

            finetuning_args.additional_target = module_names
            logger.warning_rank0("Vocab has been resized, add {} to trainable params.".format(",".join(module_names)))

        peft_kwargs = {
            "r": finetuning_args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": finetuning_args.lora_alpha,
            "lora_dropout": finetuning_args.lora_dropout,
            "modules_to_save": finetuning_args.additional_target,
        }

        if finetuning_args.pissa_init:
            if finetuning_args.pissa_iter == -1:
                logger.info_rank0("Using PiSSA initialization.")
                peft_kwargs["init_lora_weights"] = "pissa"
            else:
                logger.info_rank0(f"Using PiSSA initialization with FSVD steps {finetuning_args.pissa_iter}.")
                peft_kwargs["init_lora_weights"] = f"pissa_niter_{finetuning_args.pissa_iter}"

        adamole_config = AdaMoleConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            num_experts=finetuning_args.adamole_num_experts,
            max_threshold=finetuning_args.adamole_max_threshold,
            debug_mode=finetuning_args.adamole_debug_mode,
            **peft_kwargs,
        )
        model = get_peft_model(model, adamole_config)

    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model


def _setup_mola_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    if finetuning_args.use_dora or finetuning_args.use_rslora:
        raise ValueError("MoLA currently does not support DoRA or rank-stabilized LoRA weights.")
    if model_args.use_unsloth or model_args.use_kt:
        raise ValueError("MoLA is not compatible with KTransformers or Unsloth.")

    if is_trainable:
        logger.info_rank0("Fine-tuning method: MoLA")

    adapter_to_resume = None

    if model_args.adapter_name_or_path is not None:
        is_mergeable = True
        if getattr(model, "quantization_method", None):
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False

        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
        else:
            adapter_to_merge = model_args.adapter_name_or_path

        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "offload_folder": model_args.offload_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }
        for adapter in adapter_to_merge:
            model: "LoraModel" = PeftModel.from_pretrained(model, adapter, **init_kwargs)
            model = model.merge_and_unload()

        if len(adapter_to_merge) > 0:
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")

        if adapter_to_resume is not None:
            model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable, **init_kwargs)

        logger.info_rank0("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    if is_trainable and adapter_to_resume is None:
        if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
        else:
            target_modules = finetuning_args.lora_target

        if finetuning_args.use_llama_pro:
            target_modules = find_expanded_modules(model, target_modules, finetuning_args.freeze_trainable_layers)

        target_modules = patch_target_modules(model, finetuning_args, target_modules)

        if model_args.resize_vocab and finetuning_args.additional_target is None:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            module_names = set()
            for name, module in model.named_modules():
                if module in [input_embeddings, output_embeddings]:
                    module_names.add(name.split(".")[-1])

            finetuning_args.additional_target = module_names
            logger.warning_rank0("Vocab has been resized, add {} to trainable params.".format(",".join(module_names)))

        peft_kwargs = {
            "r": finetuning_args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": finetuning_args.lora_alpha,
            "lora_dropout": finetuning_args.lora_dropout,
            "modules_to_save": finetuning_args.additional_target,
        }

        if finetuning_args.pissa_init:
            if finetuning_args.pissa_iter == -1:
                logger.info_rank0("Using PiSSA initialization.")
                peft_kwargs["init_lora_weights"] = "pissa"
            else:
                logger.info_rank0(f"Using PiSSA initialization with FSVD steps {finetuning_args.pissa_iter}.")
                peft_kwargs["init_lora_weights"] = f"pissa_niter_{finetuning_args.pissa_iter}"

        mola_config = MolaConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            mola_num_experts=finetuning_args.mola_num_experts,
            mola_top_k=finetuning_args.mola_top_k,
            mola_use_null_expert=finetuning_args.mola_use_null_expert,
            mola_router_aux_loss_coef=finetuning_args.mola_router_aux_loss_coef,
            mola_null_expert_penalty=finetuning_args.mola_null_expert_penalty,
            mola_aux_loss_annealing=finetuning_args.mola_aux_loss_annealing,
            mola_debug_mode=finetuning_args.mola_debug_mode,
            **peft_kwargs,
        )
        model = get_peft_model(model, mola_config)

    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model


def _setup_moelpr_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    if model_args.use_unsloth or model_args.use_kt:
        raise ValueError("MoE-LPR is not compatible with KTransformers or Unsloth.")

    if is_trainable:
        logger.info_rank0(f"Fine-tuning method: MoE-LPR (Stage {finetuning_args.moelpr_stage})")

    if hasattr(config, "text_config"):
        text_config = config.text_config
    else:
        text_config = config

    num_layers = (
        getattr(text_config, "num_hidden_layers", None)
        or getattr(text_config, "num_layers", None)
        or getattr(text_config, "n_layer", None)
    )
    if num_layers is None or num_layers <= 0:
        raise ValueError("Current model does not expose a valid `num_hidden_layers` for MoE-LPR.")

    if finetuning_args.moelpr_layers_to_transform is None:
        layer_ids = list(range(num_layers))
    elif isinstance(finetuning_args.moelpr_layers_to_transform, str):
        if finetuning_args.moelpr_layers_to_transform.strip().lower() == "all":
            layer_ids = list(range(num_layers))
        else:
            layer_ids = [int(x) for x in finetuning_args.moelpr_layers_to_transform.split(",") if x.strip()]
    else:
        layer_ids = list(finetuning_args.moelpr_layers_to_transform)

    moelpr_config = MoelprConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        num_experts=finetuning_args.moelpr_num_experts,
        topk=finetuning_args.moelpr_top_k,
        layers_to_transform=layer_ids,
        aux_loss_coef=finetuning_args.moelpr_aux_loss_coef,
        lpr_loss_coef=finetuning_args.moelpr_lpr_loss_coef,
        stage=finetuning_args.moelpr_stage,
        moelpr_debug_mode=finetuning_args.moelpr_debug_mode,
    )
    model = get_peft_model(model, moelpr_config)

    if is_trainable and finetuning_args.moelpr_stage == 2:
        trainable_params = 0
        frozen_params = 0
        for name, param in model.named_parameters():
            if "moe_router_embedding" in name:
                param.requires_grad_(True)
                trainable_params += param.numel()
            else:
                param.requires_grad_(False)
                frozen_params += param.numel()
        logger.info_rank0(
            "[MoE-LPR] Stage 2 router-only training enabled: %d params trainable, %d params frozen.",
            trainable_params,
            frozen_params,
        )

    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model


def init_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""Initialize the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    if is_trainable and getattr(model, "quantization_method", None) is not None:
        if finetuning_args.finetuning_type not in ["lora", "oft"]:
            raise ValueError("Quantized models can only be used for the LoRA or OFT tuning.")

        if finetuning_args.pissa_init:
            raise ValueError("Cannot initialize PiSSA adapter on quantized models.")

    # cast trainable parameters to float32 if:
    # 1. is_trainable and not pure_bf16 and not badam and quantization_bit is not None (qlora)
    # 2. is_trainable and not pure_bf16 and not badam and not zero3 (zero3 already in fp32)
    cast_trainable_params_to_fp32 = False
    if not is_trainable:
        pass
    elif finetuning_args.pure_bf16 or finetuning_args.use_badam:
        logger.info_rank0("Pure bf16 / BAdam detected, remaining trainable params in half precision.")
    elif model_args.quantization_bit is None and (is_deepspeed_zero3_enabled() or _accelerate_wants_fsdp()):
        logger.info_rank0("DeepSpeed ZeRO3 / FSDP detected, remaining trainable params in float32.")
    else:
        logger.info_rank0("Upcasting trainable params to float32.")
        cast_trainable_params_to_fp32 = True

    if finetuning_args.finetuning_type == "full":
        _setup_full_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "freeze":
        _setup_freeze_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type in ["lora", "oft"]:
        model = _setup_lora_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    elif finetuning_args.finetuning_type == "cola":
        model = _setup_cola_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    elif finetuning_args.finetuning_type == "hydralora":
        model = _setup_hydralora_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    elif finetuning_args.finetuning_type == "adamole":
        model = _setup_adamole_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    elif finetuning_args.finetuning_type == "mola":
        model = _setup_mola_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    elif finetuning_args.finetuning_type == "moelpr":
        model = _setup_moelpr_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    else:
        raise NotImplementedError(f"Unknown finetuning type: {finetuning_args.finetuning_type}.")

    if (
        is_trainable
        and (is_deepspeed_zero3_enabled() or _accelerate_wants_fsdp())
        and getattr(model_args, "compute_dtype", None) in (torch.float16, torch.bfloat16)
    ):
        target_dtype = model_args.compute_dtype
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            if param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)
        logger.info_rank0(f"FSDP/ZeRO-3 detected, casting trainable params to {target_dtype}.")

    return model
