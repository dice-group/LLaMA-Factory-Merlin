#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from peft.metrics import pop_tracked_metrics
from peft.utils.save_and_load import get_peft_model_state_dict
from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer


@dataclass(frozen=True)
class ApproachSpec:
    train_overrides: dict[str, Any]
    required_trainable_substrings: tuple[str, ...]
    required_checkpoint_substrings: tuple[str, ...]
    metric_prefix: str


APPROACH_SPECS: dict[str, ApproachSpec] = {
    "movlora": ApproachSpec(
        train_overrides={
            "finetuning_type": "movlora",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target": "q_proj,k_proj,v_proj,o_proj",
            "use_rslora": False,
            "movlora_num_experts": 8,
            "movlora_top_k": 0,
            "movlora_router_temperature": 1.0,
            "movlora_router_jitter_noise": 0.0,
            "movlora_router_bias": False,
            "movlora_router_init_std": 0.02,
            "movlora_router_ignore_padding_tokens": False,
        },
        required_trainable_substrings=("lora_A", "lora_B", "lora_router"),
        required_checkpoint_substrings=("lora_A", "lora_B", "lora_router"),
        metric_prefix="movlora/",
    ),
    "hydralora": ApproachSpec(
        train_overrides={
            "finetuning_type": "hydralora",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target": "q_proj,k_proj,v_proj,o_proj",
            "lora_num": 3,
            "use_hydralora_experts": True,
            "hydralora_num_experts": 3,
            "hydralora_top_k": 1,
            # Enables router metric tracking in Hydra internals.
            "language_prior_weight": 0.01,
        },
        required_trainable_substrings=("lora_A", "lora_B", "lora_route", ".router."),
        required_checkpoint_substrings=("lora_A", "lora_B", "lora_route", ".router."),
        metric_prefix="hydralora/",
    ),
    "cola": ApproachSpec(
        train_overrides={
            "finetuning_type": "cola",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target": "q_proj,k_proj,v_proj,o_proj",
            "num_A": 1,
            "num_B": 3,
            "cola_strategy": "fully",
            "use_cola_experts": True,
            "cola_num_experts": 3,
            "cola_top_k": 1,
            # Enables router metric tracking in CoLA internals.
            "language_prior_weight": 0.01,
        },
        required_trainable_substrings=("lora_A", "lora_B", ".router."),
        required_checkpoint_substrings=("lora_A", "lora_B", ".router."),
        metric_prefix="cola/",
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_model_name_or_path() -> str:
    local_snapshot = Path(
        "/home/dontspamtomuch/.cache/huggingface/hub/"
        "models--hf-internal-testing--tiny-random-LlamaForCausalLM/"
        "snapshots/9fb191250dd56d0ba7ec9785a025ed29c03d5998"
    )
    if local_snapshot.exists():
        return str(local_snapshot)
    return "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _default_tokenized_path() -> str | None:
    candidate = _repo_root().parent / "tokenized_smoke" / "llama_tiny" / "moe_compare_tiny"
    if candidate.exists():
        return str(candidate)
    return None


def _parse_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _parse_kv(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --train-arg '{item}'. Expected KEY=VALUE.")
        key, raw = item.split("=", maxsplit=1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --train-arg '{item}': missing KEY.")
        parsed[key] = _parse_value(raw.strip())
    return parsed


def _load_adapter_state(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    safetensors_path = checkpoint_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))

    bin_path = checkpoint_dir / "adapter_model.bin"
    if bin_path.exists():
        try:
            return torch.load(str(bin_path), map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(str(bin_path), map_location="cpu")

    raise FileNotFoundError(f"No adapter weights found in {checkpoint_dir}")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_dataset_examples(train_dataset: Any, vocab_size: int) -> dict[str, Any]:
    _assert(len(train_dataset) > 0, "Training dataset is empty.")
    inspect_count = min(len(train_dataset), 3)
    supervised_tokens = 0
    for idx in range(inspect_count):
        sample = train_dataset[idx]
        _assert("input_ids" in sample and "labels" in sample, f"Sample {idx} missing input_ids/labels.")
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        attn = sample.get("attention_mask")

        _assert(len(input_ids) == len(labels), f"Sample {idx} has mismatched input_ids/labels length.")
        if attn is not None:
            _assert(len(attn) == len(input_ids), f"Sample {idx} has mismatched attention_mask length.")
            _assert(all(token in (0, 1) for token in attn), f"Sample {idx} attention_mask has non-binary values.")

        for token_id in input_ids:
            _assert(0 <= int(token_id) < vocab_size, f"Sample {idx} has out-of-range input token id {token_id}.")
        for label in labels:
            label = int(label)
            _assert(
                label == IGNORE_INDEX or (0 <= label < vocab_size),
                f"Sample {idx} has invalid label id {label}.",
            )
            if label != IGNORE_INDEX:
                supervised_tokens += 1

    _assert(supervised_tokens > 0, "No supervised labels found in inspected samples.")
    return {"inspected_examples": inspect_count, "supervised_tokens": supervised_tokens}


def _check_batch(batch: dict[str, torch.Tensor], pad_token_id: int, ignore_pad_for_loss: bool) -> dict[str, Any]:
    _assert("input_ids" in batch and "labels" in batch and "attention_mask" in batch, "Batch missing key tensors.")
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    _assert(input_ids.shape == labels.shape, "Batch input_ids/labels shapes do not match.")
    _assert(attention_mask.shape == input_ids.shape, "Batch attention_mask shape does not match input_ids.")

    pad_positions = attention_mask == 0
    if bool(pad_positions.any().item()):
        _assert(
            bool((input_ids[pad_positions] == pad_token_id).all().item()),
            "Found padded positions whose input_ids are not pad_token_id.",
        )
        expected_pad_label = IGNORE_INDEX if ignore_pad_for_loss else pad_token_id
        _assert(
            bool((labels[pad_positions] == expected_pad_label).all().item()),
            "Found padded positions with incorrect labels.",
        )

    non_ignored = labels != IGNORE_INDEX
    _assert(bool(non_ignored.any().item()), "Batch has no supervised (non-ignored) labels.")
    return {
        "batch_shape": tuple(input_ids.shape),
        "num_supervised_labels": int(non_ignored.sum().item()),
        "num_padded_tokens": int(pad_positions.sum().item()),
    }


def _check_approach_wiring(approach: str, model: torch.nn.Module, finetuning_args: Any) -> dict[str, Any]:
    modules = list(model.modules())
    summary: dict[str, Any] = {"checked_layers": 0}

    if approach == "movlora":
        matched = 0
        for module in modules:
            if not (hasattr(module, "lora_router") and hasattr(module, "num_experts")):
                continue
            if not isinstance(getattr(module, "num_experts", None), dict):
                continue
            if "default" not in module.num_experts:
                continue
            matched += 1
            _assert(
                int(module.num_experts["default"]) == int(finetuning_args.movlora_num_experts),
                "MoV-LoRA num_experts mismatch in instantiated layer.",
            )
            _assert(
                int(module.router_top_k["default"]) == int(finetuning_args.movlora_top_k),
                "MoV-LoRA top_k mismatch in instantiated layer.",
            )
            _assert(
                math.isclose(float(module.router_temperature["default"]), float(finetuning_args.movlora_router_temperature)),
                "MoV-LoRA router temperature mismatch in instantiated layer.",
            )
        _assert(matched > 0, "No MoV-LoRA layers found in model.")
        summary["checked_layers"] = matched
        return summary

    if approach == "hydralora":
        matched = 0
        for module in modules:
            if not hasattr(module, "use_hydralora_experts"):
                continue
            if not bool(getattr(module, "use_hydralora_experts", False)):
                continue
            matched += 1
            _assert(
                int(getattr(module, "num_experts")) == int(finetuning_args.hydralora_num_experts),
                "HydraLoRA num_experts mismatch in instantiated layer.",
            )
            _assert(
                int(getattr(module, "top_k")) == int(finetuning_args.hydralora_top_k),
                "HydraLoRA top_k mismatch in instantiated layer.",
            )
        _assert(matched > 0, "No HydraLoRA expert layers found in model.")
        summary["checked_layers"] = matched
        return summary

    if approach == "cola":
        matched = 0
        for module in modules:
            if not hasattr(module, "use_cola_experts"):
                continue
            if not bool(getattr(module, "use_cola_experts", False)):
                continue
            matched += 1
            _assert(
                int(getattr(module, "num_experts")) == int(finetuning_args.cola_num_experts),
                "CoLA num_experts mismatch in instantiated layer.",
            )
            _assert(
                int(getattr(module, "top_k")) == int(finetuning_args.cola_top_k),
                "CoLA top_k mismatch in instantiated layer.",
            )
        _assert(matched > 0, "No CoLA expert layers found in model.")
        summary["checked_layers"] = matched
        return summary

    raise ValueError(f"Unsupported approach: {approach}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quick end-to-end adapter pipeline verifier for MoE-style PEFT tuners."
    )
    parser.add_argument("--approach", choices=sorted(APPROACH_SPECS.keys()), required=True)
    parser.add_argument("--model-name-or-path", default=_default_model_name_or_path())
    parser.add_argument("--tokenized-path", default=_default_tokenized_path())
    parser.add_argument("--dataset", default="moe_compare_tiny")
    parser.add_argument("--dataset-dir", default=str(_repo_root() / "data"))
    parser.add_argument("--template", default="llama3")
    parser.add_argument("--cutoff-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2, help="Number of optimizer steps to run (minimum 2).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--offline", action="store_true", help="Set local_files_only=True to avoid network usage.")
    parser.add_argument("--output-root", default=str(_repo_root() / "tmp" / "adapter_pipeline_verify"))
    parser.add_argument("--keep-output", action="store_true", help="Keep existing output directory if present.")
    parser.add_argument("--report-json", default=None)
    parser.add_argument(
        "--train-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra training arg overrides (repeatable).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    spec = APPROACH_SPECS[args.approach]
    extra_train_args = _parse_kv(args.train_arg)
    run_steps = max(2, int(args.steps))

    output_dir = Path(args.output_root).resolve() / args.approach
    if output_dir.exists() and not args.keep_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_args: dict[str, Any] = {
        "model_name_or_path": args.model_name_or_path,
        "stage": "sft",
        "do_train": True,
        "template": args.template,
        "cutoff_len": int(args.cutoff_len),
        "overwrite_output_dir": True,
        "output_dir": str(output_dir),
        "report_to": "none",
        "save_steps": 1,
        "logging_steps": 1,
        "max_steps": run_steps,
        "per_device_train_batch_size": int(args.batch_size),
        "per_device_eval_batch_size": int(args.batch_size),
        "disable_gradient_checkpointing": True,
        "resize_vocab": True,
        "fp16": False,
        "bf16": False,
        "dataset": args.dataset,
        "dataset_dir": args.dataset_dir,
    }
    if args.tokenized_path:
        run_args["tokenized_path"] = args.tokenized_path
    if args.device == "cpu":
        run_args["use_cpu"] = True
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    run_args.update(spec.train_overrides)
    run_args.update(extra_train_args)

    # Reset metric accumulators from any prior run in this process.
    _ = pop_tracked_metrics()

    model_args, data_args, training_args, finetuning_args, _ = get_train_args(run_args)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="sft",
        **tokenizer_module,
    )
    train_dataset = dataset_module["train_dataset"]

    dataset_check = _check_dataset_examples(train_dataset, vocab_size=len(tokenizer))

    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=True)
    model.train()
    model_device = next(model.parameters()).device

    collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    dataloader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=False, collate_fn=collator)
    batch_example = next(iter(dataloader))

    pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    if pad_token_id < 0:
        pad_token_id = 0
    batch_check = _check_batch(batch_example, pad_token_id=pad_token_id, ignore_pad_for_loss=data_args.ignore_pad_token_for_loss)

    wiring_check = _check_approach_wiring(args.approach, model, finetuning_args)

    all_named_params = list(model.named_parameters())
    trainable_names = [name for name, param in all_named_params if param.requires_grad]
    required_substrings = spec.required_trainable_substrings
    for substring in required_substrings:
        _assert(
            any(substring in name for name in trainable_names),
            f"No trainable parameter found for required substring '{substring}'.",
        )
        all_hits = [(name, param) for name, param in all_named_params if substring in name]
        if all_hits:
            _assert(
                all(param.requires_grad for _, param in all_hits),
                f"Found non-trainable parameter matching required substring '{substring}'.",
            )

    named_param_dict = dict(all_named_params)
    tracked_param_names = [
        name
        for name, _ in all_named_params
        if any(sub in name for sub in required_substrings)
    ]
    tracked_initial = {
        name: named_param_dict[name].detach().cpu().clone()
        for name in tracked_param_names
    }

    optimizer = torch.optim.AdamW(
        [param for _, param in all_named_params if param.requires_grad],
        lr=float(args.learning_rate),
    )

    step_losses: list[float] = []
    metric_history: list[dict[str, float]] = []
    metric_keys_seen: set[str] = set()
    grad_seen = {substring: False for substring in required_substrings}

    ckpt1_dir = output_dir / "verify_ckpt_1"
    ckpt2_dir = output_dir / "verify_ckpt_2"

    for step_idx, batch in zip(range(run_steps), itertools.cycle(dataloader)):
        batch = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in batch.items()}
        forward_inputs = {k: v for k, v in batch.items() if k in {"input_ids", "attention_mask", "labels", "language_ids"}}

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**forward_inputs)
        loss = outputs.loss
        _assert(torch.isfinite(loss).item(), f"Encountered non-finite loss at step {step_idx}: {loss.item()}")
        loss.backward()

        for name, param in all_named_params:
            if param.grad is None:
                continue
            for substring in required_substrings:
                if substring in name:
                    grad_seen[substring] = True

        optimizer.step()
        step_losses.append(float(loss.item()))

        step_metrics = pop_tracked_metrics()
        for key, value in step_metrics.items():
            if value is None:
                continue
            _assert(math.isfinite(float(value)), f"Metric {key} has non-finite value {value}.")
            metric_keys_seen.add(key)
        metric_history.append(step_metrics)

        if step_idx == 0:
            model.save_pretrained(str(ckpt1_dir), safe_serialization=True)

    model.save_pretrained(str(ckpt2_dir), safe_serialization=True)

    for substring, seen in grad_seen.items():
        _assert(seen, f"No gradient observed for required substring '{substring}'.")

    tracked_final = {
        name: named_param_dict[name].detach().cpu()
        for name in tracked_param_names
    }
    changed_params = [
        name
        for name in tracked_param_names
        if not torch.equal(tracked_initial[name], tracked_final[name])
    ]
    _assert(len(changed_params) > 0, "No tracked adapter/router parameter changed after optimizer steps.")
    for substring in required_substrings:
        _assert(
            any(substring in name for name in changed_params),
            f"No changed parameter found for required substring '{substring}'.",
        )

    ckpt_state_1 = _load_adapter_state(ckpt1_dir)
    ckpt_state_2 = _load_adapter_state(ckpt2_dir)
    _assert(len(ckpt_state_1) > 0, "Checkpoint 1 contains no adapter tensors.")
    _assert(len(ckpt_state_2) > 0, "Checkpoint 2 contains no adapter tensors.")

    for substring in spec.required_checkpoint_substrings:
        matching = [key for key in ckpt_state_1 if substring in key]
        _assert(matching, f"Checkpoint missing required adapter key substring '{substring}'.")
        _assert(
            any(key in ckpt_state_2 and not torch.equal(ckpt_state_1[key], ckpt_state_2[key]) for key in matching),
            f"Checkpoint tensors for '{substring}' did not change between saves.",
        )

    peft_state = get_peft_model_state_dict(model)
    for substring in spec.required_checkpoint_substrings:
        _assert(
            any(substring in key for key in peft_state.keys()),
            f"get_peft_model_state_dict missing required key substring '{substring}'.",
        )

    metric_prefix_hits = sorted(key for key in metric_keys_seen if key.startswith(spec.metric_prefix))
    _assert(
        len(metric_prefix_hits) > 0,
        (
            f"No tracked metrics found for prefix '{spec.metric_prefix}'. "
            "If this is intentional, adjust the verification profile."
        ),
    )

    report = {
        "status": "ok",
        "approach": args.approach,
        "device": str(model_device),
        "model_name_or_path": model_args.model_name_or_path,
        "tokenized_path": data_args.tokenized_path,
        "dataset_checks": dataset_check,
        "batch_checks": batch_check,
        "wiring_checks": wiring_check,
        "trainable": {
            "num_trainable_params": len(trainable_names),
            "required_substrings": list(required_substrings),
        },
        "optimization": {
            "steps": run_steps,
            "losses": step_losses,
            "changed_tracked_params": len(changed_params),
            "grad_seen": grad_seen,
        },
        "metrics": {
            "prefix": spec.metric_prefix,
            "keys_seen": sorted(metric_keys_seen),
            "prefix_keys_seen": metric_prefix_hits,
            "num_metric_events": len(metric_history),
        },
        "checkpoints": {
            "checkpoint_1": str(ckpt1_dir),
            "checkpoint_2": str(ckpt2_dir),
            "num_tensors_ckpt1": len(ckpt_state_1),
            "num_tensors_ckpt2": len(ckpt_state_2),
            "required_substrings": list(spec.required_checkpoint_substrings),
        },
        "resolved_run_args": run_args,
    }

    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
