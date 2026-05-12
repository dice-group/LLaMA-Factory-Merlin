# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from peft.metrics import (
    clear_tracked_metrics,
    pop_tracked_metrics,
    record_cola_metrics,
    record_hala_metrics,
    record_hydralora_metrics,
)
from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        self._hmora_task_input_batches: List[torch.Tensor] = []
        self._hmora_task_attention_batches: List[torch.Tensor] = []
        self._hmora_task_id_batches: List[torch.Tensor] = []
        self._hmora_aux_loss_log_values: List[torch.Tensor] = []
        self._adamole_aux_loss_log_values: List[torch.Tensor] = []

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

    def _router_metrics_enabled(self) -> bool:
        value = getattr(self.finetuning_args, "track_router_metrics", None)
        if value is not None:
            return bool(value)
        return bool(
            (getattr(self.finetuning_args, "language_prior_weight", 0.0) or 0.0) > 0
            or (getattr(self.finetuning_args, "hala_balance_loss_coef", 0.0) or 0.0) > 0
            or getattr(self.finetuning_args, "language_router_mode", None) == "hard"
        )

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(
        self, model: "torch.nn.Module", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", Any]]:
        language_loss_weight = inputs.pop("language_loss_weight", None)
        self._inject_language_router_inputs(model, inputs)
        self._inject_hmora_task_inputs(model, inputs)

        if (
            self.finetuning_args.finetuning_type == "moelpr"
            and self.finetuning_args.moelpr_stage == 2
            and "lang_mask" not in inputs
        ):
            mask = self._maybe_build_moelpr_mask(inputs)
            if mask is not None:
                inputs["lang_mask"] = mask

        if getattr(self.finetuning_args, "use_language_loss_weights", False):
            base = self._compute_weighted_supervised_loss(
                model,
                inputs,
                language_loss_weight=language_loss_weight,
                return_outputs=return_outputs,
                **kwargs,
            )
        else:
            base = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        base_loss = base[0] if return_outputs else base
        extra_losses = []

        language_loss = self._compute_language_prior_loss()
        if language_loss is not None:
            extra_losses.append(language_loss)

        hala_balance_loss = self._compute_hala_balance_loss()
        if hala_balance_loss is not None:
            extra_losses.append(hala_balance_loss)

        adamole_loss = self._compute_adamole_aux_loss(model)
        if adamole_loss is not None:
            extra_losses.append(adamole_loss)

        mola_loss = self._compute_mola_aux_loss(model)
        if mola_loss is not None:
            extra_losses.append(mola_loss)

        mixlora_loss = self._compute_mixlora_aux_loss(model)
        if mixlora_loss is not None:
            extra_losses.append(mixlora_loss)

        vanilla_moelora_loss = self._compute_vanilla_moelora_aux_loss(model)
        if vanilla_moelora_loss is not None:
            extra_losses.append(vanilla_moelora_loss)

        moelpr_loss = self._compute_moelpr_aux_loss(model)
        if moelpr_loss is not None:
            extra_losses.append(moelpr_loss)

        hmora_loss = self._compute_hmora_aux_loss(model, base_loss)
        if hmora_loss is not None:
            extra_losses.append(hmora_loss)

        if not extra_losses:
            return base

        added = sum(extra_losses)
        if return_outputs:
            loss, outputs = base
            loss = loss + added
            return loss, outputs

        return base + added

    def _compute_weighted_supervised_loss(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, "torch.Tensor"],
        language_loss_weight: Optional["torch.Tensor"],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", Any]]:
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("`use_language_loss_weights` requires `labels` in the training batch.")
        if language_loss_weight is None:
            raise ValueError("`use_language_loss_weights` requires `language_loss_weight` in the training batch.")
        if labels.dim() != 2:
            raise ValueError("`use_language_loss_weights` currently expects 2D supervised-token labels.")

        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        labels = labels.to(device=logits.device)
        if logits.size(1) != labels.size(1):
            raise ValueError("`logits` and `labels` must have matching sequence length for weighted SFT loss.")

        weights = language_loss_weight.to(device=labels.device, dtype=logits.dtype).view(-1)
        if weights.numel() != labels.size(0):
            raise ValueError("`language_loss_weight` must contain one scalar per batch row.")
        if torch.any(weights < 0):
            raise ValueError("`language_loss_weight` values must be non-negative.")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view_as(shift_labels)
        token_mask = shift_labels.ne(IGNORE_INDEX).to(dtype=logits.dtype)
        row_weights = weights.view(-1, 1)
        normalizer = (token_mask * row_weights).sum().clamp_min(torch.finfo(logits.dtype).eps)
        loss = (token_losses * token_mask * row_weights).sum() / normalizer
        return (loss, outputs) if return_outputs else loss

    @override
    def training_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, "torch.Tensor"],
        *args,
        **kwargs,
    ) -> "torch.Tensor":
        if (
            os.environ.get("LLAMAFACTORY_DDP_STATIC_GRAPH", "").strip().lower() in {"1", "true", "yes"}
            and not getattr(self, "_ddp_static_graph_set", False)
        ):
            set_static_graph = getattr(model, "_set_static_graph", None)
            if callable(set_static_graph):
                set_static_graph()
                logger.info_rank0("[DDP] Enabled static graph mode for distributed training.")
            self._ddp_static_graph_set = True

        loss = super().training_step(model, inputs, *args, **kwargs)
        # Gradient checkpointing recomputes routed layers during backward and can
        # repopulate routed auxiliary caches after compute_loss has consumed them.
        self._flush_language_router_cache()
        self._flush_hala_balance_router_cache()

        if not self._router_metrics_enabled():
            return loss

        if self.finetuning_args.finetuning_type == "cola":
            if not hasattr(self, "_cola_router_params"):
                module = getattr(model, "module", model)
                self._cola_router_params = [
                    param for name, param in module.named_parameters() if ".router." in name
                ]
            total = len(self._cola_router_params)
            present = sum(1 for param in self._cola_router_params if param.grad is not None)
            frac = float(present / total) if total > 0 else 0.0
            record_cola_metrics(
                {
                    "router_grad_present_frac": frac,
                    "expert_router_grad_present_frac": frac,
                },
                weight=1.0,
            )
        elif self.finetuning_args.finetuning_type == "hydralora":
            if not hasattr(self, "_hydralora_expert_router_params"):
                module = getattr(model, "module", model)
                self._hydralora_expert_router_params = [
                    param for name, param in module.named_parameters() if ".router." in name
                ]
                self._hydralora_head_router_params = [
                    param for name, param in module.named_parameters() if ".lora_route." in name
                ]

            expert_total = len(self._hydralora_expert_router_params)
            expert_present = sum(1 for param in self._hydralora_expert_router_params if param.grad is not None)
            head_total = len(self._hydralora_head_router_params)
            head_present = sum(1 for param in self._hydralora_head_router_params if param.grad is not None)
            total = expert_total + head_total
            present = expert_present + head_present
            record_hydralora_metrics(
                {
                    "router_grad_present_frac": float(present / total) if total > 0 else 0.0,
                    "expert_router_grad_present_frac": float(expert_present / expert_total) if expert_total > 0 else 0.0,
                    "head_router_grad_present_frac": float(head_present / head_total) if head_total > 0 else 0.0,
                },
                weight=1.0,
            )
        elif self.finetuning_args.finetuning_type == "hala":
            if not hasattr(self, "_hala_expert_router_params"):
                module = getattr(model, "module", model)
                self._hala_expert_router_params = [
                    param for name, param in module.named_parameters() if ".router." in name
                ]
                self._hala_head_router_params = [
                    param for name, param in module.named_parameters() if ".lora_route." in name
                ]

            expert_total = len(self._hala_expert_router_params)
            expert_present = sum(1 for param in self._hala_expert_router_params if param.grad is not None)
            head_total = len(self._hala_head_router_params)
            head_present = sum(1 for param in self._hala_head_router_params if param.grad is not None)
            total = expert_total + head_total
            present = expert_present + head_present
            record_hala_metrics(
                {
                    "router_grad_present_frac": float(present / total) if total > 0 else 0.0,
                    "expert_router_grad_present_frac": float(expert_present / expert_total) if expert_total > 0 else 0.0,
                    "head_router_grad_present_frac": float(head_present / head_total) if head_total > 0 else 0.0,
                },
                weight=1.0,
            )

        return loss

    @override
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        is_train_log = any(key in logs for key in ("loss", "learning_rate")) or any(
            key.startswith("train/") for key in logs
        )
        if is_train_log and self._hmora_aux_loss_log_values:
            logs["hmora_aux_loss"] = float(torch.stack(self._hmora_aux_loss_log_values).mean().cpu())
            self._hmora_aux_loss_log_values.clear()
        if is_train_log and self._adamole_aux_loss_log_values:
            logs["adamole_aux_loss"] = float(torch.stack(self._adamole_aux_loss_log_values).mean().cpu())
            self._adamole_aux_loss_log_values.clear()

        extra = pop_tracked_metrics()
        if extra:
            phase_prefix = "train" if is_train_log else "eval"
            for key, value in extra.items():
                if value is None:
                    continue
                scoped = key if key.startswith(("train/", "eval/", "test/")) else f"{phase_prefix}/{key}"
                logs[scoped] = value
        elif not is_train_log:
            clear_tracked_metrics()
        super().log(logs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    def _compute_language_prior_loss(self) -> Optional[torch.Tensor]:
        weight = getattr(self.finetuning_args, "language_prior_weight", 0.0) or 0.0
        if weight <= 0:
            self._flush_language_router_cache()
            return None

        states = self._flush_language_router_cache()
        if not states:
            return None

        grouped_states: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        valid_target_count = 0
        for _, logits, targets in states:
            if logits is None or targets is None:
                continue
            if logits.dim() > 2:
                logits = logits.mean(dim=1)
            valid = targets >= 0
            if not valid.any():
                continue
            logits_valid = logits[valid]
            targets_valid = targets[valid]
            grouped_states.setdefault(int(logits_valid.size(-1)), []).append((logits_valid, targets_valid))
            valid_target_count += int(targets_valid.numel())

        if not grouped_states:
            return None

        raw_losses = []
        state_count = 0
        for grouped in grouped_states.values():
            logits_cat = torch.cat([item[0] for item in grouped], dim=0)
            targets_cat = torch.cat([item[1] for item in grouped], dim=0)
            losses = F.cross_entropy(logits_cat, targets_cat, reduction="none")
            offset = 0
            for logits_valid, _targets_valid in grouped:
                length = int(logits_valid.size(0))
                raw_losses.append(losses[offset : offset + length].mean())
                offset += length
            state_count += len(grouped)

        raw = torch.stack(raw_losses).mean()
        aux = weight * raw
        if self._router_metrics_enabled():
            language_prior_loss_raw = float(raw.detach().mean().cpu())
            language_prior_loss = float(aux.detach().mean().cpu())
            metrics = {
                "language_prior_loss_raw": language_prior_loss_raw,
                "language_prior_loss": language_prior_loss,
                "language_prior_router_state_count": float(state_count),
                "language_prior_valid_target_count": float(valid_target_count),
            }
            if self.finetuning_args.finetuning_type == "hydralora":
                record_hydralora_metrics(metrics, weight=1.0)
            elif self.finetuning_args.finetuning_type == "hala":
                record_hala_metrics(metrics, weight=1.0)
            else:
                record_cola_metrics(metrics, weight=1.0)
        return aux

    def _compute_adamole_aux_loss(self, model: "torch.nn.Module") -> Optional[torch.Tensor]:
        coef = getattr(self.finetuning_args, "adamole_aux_loss_coef", 0.0) or 0.0
        if self.finetuning_args.finetuning_type != "adamole" or coef <= 0:
            return None

        module = getattr(model, "module", model)
        aux_fn = getattr(module, "get_aux_loss", None)
        if not callable(aux_fn):
            return None

        aux = aux_fn()
        if aux is None:
            return None

        scaled = coef * aux
        self._adamole_aux_loss_log_values.append(scaled.detach().mean())
        return scaled

    def _compute_mola_aux_loss(self, model: "torch.nn.Module") -> Optional[torch.Tensor]:
        if self.finetuning_args.finetuning_type != "mola":
            return None

        coef = float(getattr(self.finetuning_args, "mola_router_aux_loss_coef", 0.0) or 0.0)
        if coef <= 0:
            return None

        module = getattr(model, "module", model)
        aux_fn = getattr(module, "get_aux_loss", None)
        if not callable(aux_fn):
            return None

        aux = aux_fn()
        if aux is None:
            return None

        if self.finetuning_args.mola_aux_loss_annealing:
            end_coef = getattr(self.finetuning_args, "mola_aux_loss_coef_end", None)
            if end_coef is None:
                end_coef = 0.0
            max_steps = getattr(self.state, "max_steps", 0) or 0
            if max_steps > 0:
                progress = min(1.0, float(self.state.global_step) / float(max_steps))
            else:
                progress = 1.0
            coef = (1.0 - progress) * coef + progress * float(end_coef)

        scaled = coef * aux
        self.log({"mola_aux_loss": float(scaled.detach().mean().cpu())})
        return scaled

    def _compute_mixlora_aux_loss(self, model: "torch.nn.Module") -> Optional[torch.Tensor]:
        if self.finetuning_args.finetuning_type != "mixlora":
            return None

        coef = float(getattr(self.finetuning_args, "mixlora_router_aux_loss_coef", 0.0) or 0.0)
        if coef <= 0:
            return None

        module = getattr(model, "module", model)
        aux_fn = getattr(module, "get_aux_loss", None)
        if not callable(aux_fn):
            return None

        aux = aux_fn()
        if aux is None:
            return None

        scaled = coef * aux
        self.log({"mixlora_aux_loss": float(scaled.detach().mean().cpu())})
        return scaled

    def _compute_vanilla_moelora_aux_loss(self, model: "torch.nn.Module") -> Optional[torch.Tensor]:
        if self.finetuning_args.finetuning_type != "vanilla_moelora":
            return None

        coef = float(getattr(self.finetuning_args, "vanilla_moelora_router_aux_loss_coef", 0.0) or 0.0)
        if coef <= 0:
            return None

        module = getattr(model, "module", model)
        aux_fn = getattr(module, "get_aux_loss", None)
        if not callable(aux_fn):
            return None

        aux = aux_fn()
        if aux is None:
            return None

        scaled = coef * aux
        self.log({"vanilla_moelora_aux_loss": float(scaled.detach().mean().cpu())})
        return scaled

    def _compute_moelpr_aux_loss(self, model: "torch.nn.Module") -> Optional[torch.Tensor]:
        if self.finetuning_args.finetuning_type != "moelpr":
            return None

        module = getattr(model, "module", model)
        aux_fn = getattr(module, "get_aux_loss", None)
        if not callable(aux_fn):
            return None

        aux = aux_fn()
        if aux is None:
            return None

        self.log({"moelpr_aux_loss": float(aux.detach().mean().cpu())})
        return aux

    def _compute_hala_balance_loss(self) -> Optional[torch.Tensor]:
        if self.finetuning_args.finetuning_type != "hala":
            self._flush_hala_balance_router_cache()
            return None

        coef = float(getattr(self.finetuning_args, "hala_balance_loss_coef", 0.0) or 0.0)
        kind = getattr(self.finetuning_args, "hala_balance_loss_kind", "none")
        target = getattr(self.finetuning_args, "hala_balance_target", "expert")
        if coef <= 0 or kind == "none":
            self._flush_hala_balance_router_cache()
            return None
        if kind not in {"uniform_importance", "target_distribution_importance"} or target != "expert":
            raise ValueError(
                "Unsupported HALA balance objective; use uniform_importance/expert "
                "or target_distribution_importance/expert."
            )

        states = self._flush_hala_balance_router_cache()
        if not states:
            return None

        raw_losses = []
        token_count = 0
        for state in states:
            _name, logits, targets = (state[0], state[1], state[2] if len(state) > 2 else None)
            if logits is None or logits.numel() == 0:
                continue
            if logits.dim() > 2:
                logits = logits.mean(dim=1)
            probs = torch.softmax(logits.to(torch.float32), dim=-1)
            if probs.size(-1) <= 0:
                continue
            importance = probs.mean(dim=0)
            if kind == "target_distribution_importance":
                if targets is None or not torch.is_tensor(targets):
                    continue
                targets = targets.to(device=probs.device, dtype=torch.long).reshape(-1)
                if targets.numel() != probs.size(0):
                    continue
                valid = (targets >= 0) & (targets < probs.size(-1))
                if not valid.any():
                    continue
                counts = torch.bincount(targets[valid], minlength=probs.size(-1)).to(probs.dtype)
                target_importance = counts / counts.sum().clamp_min(1.0)
            else:
                target_importance = torch.full_like(importance, 1.0 / float(probs.size(-1)))
            raw_losses.append(F.mse_loss(importance, target_importance, reduction="mean") * float(probs.size(-1)))
            token_count += int(probs.size(0))

        if not raw_losses:
            return None

        raw = torch.stack(raw_losses).mean()
        aux = coef * raw
        if self._router_metrics_enabled():
            record_hala_metrics(
                {
                    "hala_balance_loss_raw": float(raw.detach().mean().cpu()),
                    "hala_balance_loss": float(aux.detach().mean().cpu()),
                    "hala_balance_router_state_count": float(len(raw_losses)),
                    "hala_balance_token_count": float(token_count),
                },
                weight=1.0,
            )
        return aux

    def _compute_hmora_aux_loss(
        self, model: "torch.nn.Module", base_loss: "torch.Tensor"
    ) -> Optional[torch.Tensor]:
        if self.finetuning_args.finetuning_type != "hmora":
            return None

        module = getattr(model, "module", model)
        aux_fn = getattr(module, "get_aux_loss", None)
        if not callable(aux_fn):
            return None

        lm_scale = float(getattr(self.finetuning_args, "hmora_lambda_lm", 1.0) or 1.0)
        aux_scale = float(getattr(self.finetuning_args, "hmora_lambda_auxiliary", 0.0) or 0.0)

        extra_terms = []
        if lm_scale != 1.0:
            extra_terms.append((lm_scale - 1.0) * base_loss)

        aux = aux_fn(include_task_router=True)
        if aux is not None and aux_scale > 0:
            scaled_aux = aux_scale * aux
            self._hmora_aux_loss_log_values.append(scaled_aux.detach().mean())
            extra_terms.append(scaled_aux)

        if not extra_terms:
            return None
        return sum(extra_terms)

    def _maybe_build_moelpr_mask(self, inputs: Dict[str, "torch.Tensor"]) -> Optional["torch.Tensor"]:
        target_id = getattr(self.finetuning_args, "moelpr_target_language_id", None)
        if target_id is None:
            return None
        lang_ids = inputs.get("language_ids")
        input_ids = inputs.get("input_ids")
        if lang_ids is None or input_ids is None:
            return None
        if lang_ids.dim() == 1:
            lang_ids = lang_ids.unsqueeze(1)
        seq_len = input_ids.size(1)
        mask = (lang_ids == target_id).to(dtype=torch.bool)
        if mask.size(1) == 1:
            mask = mask.expand(-1, seq_len)
        return mask

    def _inject_language_router_inputs(self, model: "torch.nn.Module", inputs: Dict[str, "torch.Tensor"]) -> None:
        if self.finetuning_args.finetuning_type not in {"cola", "hydralora", "hala", "soft_moe", "grad_iso", "lang_gate"}:
            return

        language_ids = inputs.get("language_ids")
        if self.finetuning_args.finetuning_type in {"hala", "soft_moe", "grad_iso", "lang_gate"} and language_ids is None:
            raise ValueError(f"{self.finetuning_args.finetuning_type} training requires tokenized batches to contain language_ids.")
        module = getattr(model, "module", model)
        routed_modules = getattr(self, "_language_routed_modules", None)
        if routed_modules is None:
            routed_modules = [
                submodule
                for submodule in module.modules()
                if hasattr(submodule, "language_guidance_scope") and hasattr(submodule, "base_layer")
            ]
            self._language_routed_modules = routed_modules
            if routed_modules:
                logger.info_rank0(
                    f"[LPR] Found {len(routed_modules)} language-routed adapter layers; injecting language_ids per batch."
                )

        if language_ids is None:
            for routed_module in routed_modules:
                setattr(routed_module, "language_ids", None)
            return

        for routed_module in routed_modules:
            setattr(routed_module, "language_ids", language_ids)

    def _build_hmora_task_batch(
        self, inputs: Dict[str, "torch.Tensor"]
    ) -> tuple[Optional["torch.Tensor"], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        if input_ids is None or attention_mask is None:
            return None, None, None

        labels = inputs.get("labels")
        pad_token_id = getattr(self.processing_class, "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0

        if labels is None:
            task_ids = inputs.get("task_ids")
            if task_ids is None:
                task_ids = inputs.get("language_ids")
            return input_ids, attention_mask, task_ids

        prompt_masks = []
        max_prompt_len = 0
        for row_labels, row_attention in zip(labels, attention_mask):
            prompt_mask = (row_attention > 0) & (row_labels == IGNORE_INDEX)
            if not torch.any(prompt_mask):
                prompt_mask = row_attention > 0
            prompt_masks.append(prompt_mask)
            max_prompt_len = max(max_prompt_len, int(prompt_mask.sum().item()))

        if max_prompt_len <= 0:
            task_ids = inputs.get("task_ids")
            if task_ids is None:
                task_ids = inputs.get("language_ids")
            return input_ids, attention_mask, task_ids

        task_input_ids = input_ids.new_full((input_ids.size(0), max_prompt_len), int(pad_token_id))
        task_attention_mask = attention_mask.new_zeros((attention_mask.size(0), max_prompt_len))
        for row_idx, prompt_mask in enumerate(prompt_masks):
            selected = input_ids[row_idx][prompt_mask]
            length = selected.size(0)
            if length == 0:
                continue
            task_input_ids[row_idx, :length] = selected
            task_attention_mask[row_idx, :length] = 1

        task_ids = inputs.get("task_ids")
        if task_ids is None:
            task_ids = inputs.get("language_ids")
        return task_input_ids, task_attention_mask, task_ids

    def _inject_hmora_task_inputs(self, model: "torch.nn.Module", inputs: Dict[str, "torch.Tensor"]) -> None:
        if self.finetuning_args.finetuning_type != "hmora":
            return

        module = getattr(model, "module", model)
        set_task_inputs = getattr(module, "set_runtime_task_inputs", None)
        if not callable(set_task_inputs):
            return

        task_input_ids, task_attention_mask, task_ids = self._build_hmora_task_batch(inputs)
        set_task_inputs(
            task_input_ids=task_input_ids,
            task_attention_mask=task_attention_mask,
            task_ids=task_ids,
        )

        if (
            self.model.training
            and self.finetuning_args.hmora_use_div_loss
            and task_input_ids is not None
            and task_attention_mask is not None
        ):
            self._hmora_task_input_batches.append(task_input_ids.detach().cpu())
            self._hmora_task_attention_batches.append(task_attention_mask.detach().cpu())
            if task_ids is not None:
                self._hmora_task_id_batches.append(task_ids.detach().cpu())

    def _clear_hmora_task_batch_cache(self) -> None:
        self._hmora_task_input_batches.clear()
        self._hmora_task_attention_batches.clear()
        self._hmora_task_id_batches.clear()

    def _apply_hmora_task_router_step_loss(self, model: "torch.nn.Module") -> Optional["torch.Tensor"]:
        if not self.finetuning_args.hmora_use_div_loss:
            self._clear_hmora_task_batch_cache()
            return None
        if not self._hmora_task_input_batches or not self._hmora_task_attention_batches:
            return None

        module = getattr(model, "module", model)
        aux_fn = getattr(module, "get_task_router_aux_loss", None)
        if not callable(aux_fn):
            self._clear_hmora_task_batch_cache()
            return None

        max_task_len = max(batch.size(1) for batch in self._hmora_task_input_batches)
        pad_token_id = getattr(self.processing_class, "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0

        padded_task_inputs = []
        padded_task_masks = []
        for task_input_batch, task_attention_batch in zip(
            self._hmora_task_input_batches,
            self._hmora_task_attention_batches,
        ):
            pad_width = max_task_len - task_input_batch.size(1)
            if pad_width > 0:
                task_input_batch = F.pad(task_input_batch, (0, pad_width), value=int(pad_token_id))
                task_attention_batch = F.pad(task_attention_batch, (0, pad_width), value=0)
            padded_task_inputs.append(task_input_batch)
            padded_task_masks.append(task_attention_batch)

        task_input_ids = torch.cat(padded_task_inputs, dim=0).to(self.accelerator.device)
        task_attention_mask = torch.cat(padded_task_masks, dim=0).to(self.accelerator.device)
        task_ids = None
        if self._hmora_task_id_batches:
            task_ids = torch.cat(self._hmora_task_id_batches, dim=0).to(self.accelerator.device)

        self._clear_hmora_task_batch_cache()

        aux_scale = float(getattr(self.finetuning_args, "hmora_lambda_auxiliary", 0.0) or 0.0)
        if aux_scale <= 0:
            return None

        aux = aux_fn(task_input_ids=task_input_ids, task_attention_mask=task_attention_mask, task_ids=task_ids)
        if aux is None:
            return None

        scaled_aux = aux_scale * aux
        self.accelerator.backward(scaled_aux)
        self.log({"hmora_task_router_aux_loss": float(scaled_aux.detach().mean().cpu())})
        return scaled_aux

    def _flush_language_router_cache(self) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        caches: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        modules = getattr(self, "_language_routed_modules", None)
        if modules is None:
            module = getattr(self.model, "module", self.model)
            modules = [
                submodule
                for submodule in module.modules()
                if hasattr(submodule, "language_guidance_scope") and hasattr(submodule, "base_layer")
            ]
            self._language_routed_modules = modules
        for module in modules:
            pop_fn = getattr(module, "pop_language_router_cache", None)
            if callable(pop_fn):
                caches.extend(pop_fn())
        return caches

    def _flush_hala_balance_router_cache(self) -> list[tuple]:
        caches: list[tuple] = []
        modules = getattr(self, "_language_routed_modules", None)
        if modules is None:
            module = getattr(self.model, "module", self.model)
            modules = [
                submodule
                for submodule in module.modules()
                if hasattr(submodule, "language_guidance_scope") and hasattr(submodule, "base_layer")
            ]
            self._language_routed_modules = modules
        for module in modules:
            pop_fn = getattr(module, "pop_hala_balance_router_cache", None)
            if callable(pop_fn):
                caches.extend(pop_fn())
        return caches
