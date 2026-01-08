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

import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

import torch
import transformers
from peft import PeftModel
from transformers import PreTrainedModel, ProcessorMixin, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing_extensions import override
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

from ..extras import logging
from ..extras.constants import TRAINER_LOG, V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import get_peak_memory, is_env_enabled, use_ray
from ..extras.packages import is_safetensors_available


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


def fix_valuehead_checkpoint(
    model: "AutoModelForCausalLMWithValueHead", output_dir: str, safe_serialization: bool
) -> None:
    r"""Fix the valuehead checkpoint files.

    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    """
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    if safe_serialization:
        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
        with safe_open(path_to_checkpoint, framework="pt", device="cpu") as f:
            state_dict: dict[str, torch.Tensor] = {key: f.get_tensor(key).clone() for key in f.keys()}
    else:
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict: dict[str, torch.Tensor] = torch.load(path_to_checkpoint, map_location="cpu", weights_only=True)

    os.remove(path_to_checkpoint)
    decoder_state_dict, v_head_state_dict = {}, {}
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param

    model.pretrained_model.save_pretrained(
        output_dir, state_dict=decoder_state_dict or None, safe_serialization=safe_serialization
    )

    if safe_serialization:
        save_file(v_head_state_dict, os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
    else:
        torch.save(v_head_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))

    logger.info_rank0(f"Value head model saved at: {output_dir}")


class FixValueHeadModelCallback(TrainerCallback):
    r"""A callback for fixing the checkpoint for valuehead models."""

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            fix_valuehead_checkpoint(
                model=kwargs.pop("model"), output_dir=output_dir, safe_serialization=args.save_safetensors
            )


class SaveProcessorCallback(TrainerCallback):
    r"""A callback for saving the processor."""

    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            self.processor.save_pretrained(output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)


class SaveAdapterCheckpointCallback(TrainerCallback):
    r"""A callback for saving PEFT adapters alongside FSDP checkpoints."""

    def __init__(self, suffix: str = "_adapter") -> None:
        self.suffix = suffix
        self.accelerator = None

    def set_accelerator(self, accelerator: Any) -> None:
        self.accelerator = accelerator

    def _save_adapter(self, model: torch.nn.Module, output_dir: str, safe_serialization: bool) -> None:
        unwrapped = getattr(model, "module", model)
        if not isinstance(unwrapped, PeftModel):
            return

        adapter_dir = f"{output_dir}{self.suffix}"
        start_time = time.monotonic()

        if isinstance(model, FSDP):
            rank = 0
            world_size = None
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            logger.info_rank0("Adapter save start (FSDP): output_dir=%s adapter_dir=%s world_size=%s", output_dir, adapter_dir, world_size if world_size is not None else "n/a")
            fsdp_plugin = getattr(getattr(self.accelerator, "state", None), "fsdp_plugin", None)
            if self.accelerator is not None and fsdp_plugin is not None:
                try:
                    import inspect
                    from accelerate.utils.fsdp_utils import save_fsdp_model
                    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

                    if "adapter_only" in inspect.signature(save_fsdp_model).parameters:
                        os.makedirs(adapter_dir, exist_ok=True)
                        orig_state_dict_type = fsdp_plugin.state_dict_type
                        orig_state_dict_config = fsdp_plugin.state_dict_config
                        orig_offload = getattr(fsdp_plugin.state_dict_config, "offload_to_cpu", None)
                        orig_rank0 = getattr(fsdp_plugin.state_dict_config, "rank0_only", None)
                        try:
                            fsdp_plugin.state_dict_type = StateDictType.FULL_STATE_DICT
                            is_multi = self.accelerator.num_processes > 1
                            fsdp_plugin.state_dict_config = FullStateDictConfig(
                                offload_to_cpu=is_multi, rank0_only=is_multi
                            )
                            save_start = time.monotonic()
                            save_fsdp_model(
                                fsdp_plugin,
                                self.accelerator,
                                model,
                                adapter_dir,
                                adapter_only=True,
                            )
                            save_end = time.monotonic()
                        finally:
                            fsdp_plugin.state_dict_type = orig_state_dict_type
                            fsdp_plugin.state_dict_config = orig_state_dict_config
                            if orig_offload is not None:
                                fsdp_plugin.state_dict_config.offload_to_cpu = orig_offload
                            if orig_rank0 is not None:
                                fsdp_plugin.state_dict_config.rank0_only = orig_rank0

                        if rank == 0:
                            fsdp_path = os.path.join(adapter_dir, "pytorch_model_fsdp.bin")
                            if os.path.exists(fsdp_path):
                                load_start = time.monotonic()
                                try:
                                    adapter_state = torch.load(fsdp_path, map_location="cpu", weights_only=True)
                                except TypeError:
                                    adapter_state = torch.load(fsdp_path, map_location="cpu")
                                load_end = time.monotonic()
                                adapter_keys = len(adapter_state or {})
                                logger.info_rank0(
                                    "Adapter shard state built: shard_keys=%s adapter_keys=%s get_state_s=%.2f filter_s=%.2f",
                                    adapter_keys,
                                    adapter_keys,
                                    save_end - save_start,
                                    load_end - load_start,
                                )
                                unwrapped.save_pretrained(
                                    adapter_dir,
                                    state_dict=adapter_state,
                                    safe_serialization=safe_serialization,
                                )
                                os.remove(fsdp_path)
                                logger.info_rank0(
                                    "Adapter checkpoint saved at: %s (adapter_only fsdp total_s=%.2f)",
                                    adapter_dir,
                                    time.monotonic() - start_time,
                                )
                        if torch.distributed.is_initialized():
                            torch.distributed.barrier()
                        return
                except Exception as exc:
                    exc_msg = str(exc).replace("\n", " ").replace("\r", " ")
                    logger.warning_rank0("Adapter-only FSDP save failed; falling back to manual path: %s", exc_msg)
            try:
                from torch.distributed.checkpoint import FileSystemWriter, save as dcp_save
                from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
                from peft.utils.save_and_load import get_peft_model_state_dict

                shard_dir = f"{adapter_dir}_sharded"
                # Keep frozen params so routers/aux modules aren't dropped if requires_grad is toggled.
                opts = StateDictOptions(full_state_dict=False, cpu_offload=True, ignore_frozen_params=False)
                shard_start = time.monotonic()
                shard_state = get_model_state_dict(model, options=opts)
                shard_end = time.monotonic()
                if shard_state is not None and any(k.startswith("_fsdp_wrapped_module.") for k in shard_state):
                    shard_state = {
                        k.removeprefix("_fsdp_wrapped_module."): v for k, v in shard_state.items()
                    }
                filter_start = time.monotonic()
                adapter_state = get_peft_model_state_dict(unwrapped, state_dict=shard_state)
                filter_end = time.monotonic()
                logger.info_rank0("Adapter shard state built: shard_keys=%s adapter_keys=%s get_state_s=%.2f filter_s=%.2f", len(shard_state or {}), len(adapter_state or {}), shard_end - shard_start, filter_end - filter_start)
                writer = FileSystemWriter(shard_dir)
                save_start = time.monotonic()
                dcp_save(adapter_state, storage_writer=writer)
                save_end = time.monotonic()
                if rank == 0:
                    os.makedirs(shard_dir, exist_ok=True)
                    unwrapped.peft_config["default"].save_pretrained(shard_dir)
                    meta_path = os.path.join(shard_dir, "adapter_sharded.json")
                    with open(meta_path, "w", encoding="utf-8") as meta_file:
                        json.dump(
                            {
                                "format": "torch.distributed.checkpoint",
                                "base_model_name_or_path": unwrapped.peft_config["default"].base_model_name_or_path,
                            },
                            meta_file,
                            indent=2,
                        )
                    done_path = os.path.join(shard_dir, ".done")
                    with open(done_path, "w", encoding="utf-8") as done_file:
                        done_file.write("ok")
                    logger.info_rank0("Adapter shard checkpoint saved at: %s (dcp_save_s=%.2f total_s=%.2f)", shard_dir, save_end - save_start, save_end - start_time,)
                return
            except Exception as exc:
                exc_msg = str(exc).replace("\n", " ").replace("\r", " ")
                logger.warning_rank0("Adapter shard save failed, falling back to full state dict: %s", exc_msg)
                full_state = None
                try:
                    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

                    opts = StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True)
                    gather_start = time.monotonic()
                    full_state = get_model_state_dict(model, options=opts)
                    gather_end = time.monotonic()
                    logger.info_rank0("Full state dict gathered via DCP: keys=%s time_s=%.2f", len(full_state or {}), gather_end - gather_start)
                except Exception as inner_exc:
                    inner_msg = str(inner_exc).replace("\n", " ").replace("\r", " ")
                    logger.warning_rank0("Full state dict gather via DCP failed; using FSDP.full_state_dict: %s", inner_msg)
                    gather_start = time.monotonic()
                    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
                        full_state = model.state_dict()
                    gather_end = time.monotonic()
                    logger.info_rank0("Full state dict gathered via FSDP: keys=%s time_s=%.2f", len(full_state or {}), gather_end - gather_start)
                if full_state is not None and any(k.startswith("_fsdp_wrapped_module.") for k in full_state):
                    full_state = {
                        k.removeprefix("_fsdp_wrapped_module."): v for k, v in full_state.items()
                    }

                if rank == 0:
                    os.makedirs(adapter_dir, exist_ok=True)
                    adapter_state = None
                    try:
                        from peft.utils.save_and_load import get_peft_model_state_dict

                        filter_start = time.monotonic()
                        adapter_state = get_peft_model_state_dict(unwrapped, state_dict=full_state)
                        filter_end = time.monotonic()
                    except Exception:
                        adapter_state = full_state

                    save_start = time.monotonic()
                    unwrapped.save_pretrained(
                        adapter_dir, state_dict=adapter_state, safe_serialization=safe_serialization
                    )
                    save_end = time.monotonic()
                    logger.info_rank0("Adapter checkpoint saved at: %s (filter_s=%.2f save_s=%.2f total_s=%.2f)", adapter_dir, filter_end - filter_start if "filter_end" in locals() else 0.0, save_end - save_start, save_end - start_time)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                return

        rank = 0
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        if rank != 0:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return
        os.makedirs(adapter_dir, exist_ok=True)
        unwrapped.save_pretrained(adapter_dir, safe_serialization=safe_serialization)
        logger.info_rank0("Adapter checkpoint saved at: %s (non-FSDP total_s=%.2f)", adapter_dir, time.monotonic() - start_time)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        self._save_adapter(kwargs.pop("model"), output_dir, args.save_safetensors)


class PissaConvertCallback(TrainerCallback):
    r"""A callback for converting the PiSSA adapter to a normal one."""

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            logger.info_rank0(f"Initial PiSSA adapter will be saved at: {pissa_init_dir}.")
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_init_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            # 1. save a pissa backup with init_lora_weights: True
            # 2. save a converted lora with init_lora_weights: pissa
            # 3. load the pissa backup with init_lora_weights: True
            # 4. delete the initial adapter and change init_lora_weights to pissa
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                model.save_pretrained(
                    pissa_convert_dir,
                    safe_serialization=args.save_safetensors,
                    path_initial_model_for_weight_conversion=pissa_init_dir,
                )
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


class LogCallback(TrainerCallback):
    r"""A callback for logging training and evaluation status."""

    def __init__(self) -> None:
        # Progress
        self.start_time = 0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        # Status
        self.aborted = False
        self.do_train = False
        # Web UI
        self.webui_mode = is_env_enabled("LLAMABOARD_ENABLED")
        if self.webui_mode and not use_ray():
            signal.signal(signal.SIGABRT, self._set_abort)
            self.logger_handler = logging.LoggerHandler(os.getenv("LLAMABOARD_WORKDIR"))
            logging.add_handler(self.logger_handler)
            transformers.logging.add_handler(self.logger_handler)

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def _write_log(self, output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if (
            args.should_save
            and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
            and args.overwrite_output_dir
        ):
            logger.warning_rank0_once("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()

    @override
    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss"),
            eval_loss=state.log_history[-1].get("eval_loss"),
            predict_loss=state.log_history[-1].get("predict_loss"),
            reward=state.log_history[-1].get("reward"),
            accuracy=state.log_history[-1].get("rewards/accuracies"),
            lr=state.log_history[-1].get("learning_rate"),
            epoch=state.log_history[-1].get("epoch"),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / (time.time() - self.start_time), 2)
            logs["total_tokens"] = state.num_input_tokens_seen

        if is_env_enabled("RECORD_VRAM"):
            vram_allocated, vram_reserved = get_peak_memory()
            logs["vram_allocated"] = round(vram_allocated / (1024**3), 2)
            logs["vram_reserved"] = round(vram_reserved / (1024**3), 2)

        logs = {k: v for k, v in logs.items() if v is not None}
        if self.webui_mode and all(key in logs for key in ("loss", "lr", "epoch")):
            log_str = f"'loss': {logs['loss']:.4f}, 'learning_rate': {logs['lr']:2.4e}, 'epoch': {logs['epoch']:.2f}"
            for extra_key in ("reward", "accuracy", "throughput"):
                if logs.get(extra_key):
                    log_str += f", '{extra_key}': {logs[extra_key]:.2f}"

            logger.info_rank0("{" + log_str + "}")

        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.do_train:
            return

        if self.aborted:
            sys.exit(0)

        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)

            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time,
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)


class ReporterCallback(TrainerCallback):
    r"""A callback for reporting training status to external logger."""

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "llamafactory")

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return

        if "wandb" in args.report_to:
            import wandb

            wandb.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if self.finetuning_args.use_swanlab:
            import swanlab  # type: ignore

            swanlab.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )
