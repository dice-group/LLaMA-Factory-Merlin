import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.generic_adapter import MethodAdapter
from approaches.adamoe_1.model import MolaModel
from approaches.adamoe_1.config import MolaConfig


class MolaAdapter(MethodAdapter):
    """Adapter wrapper for MoLA"""

    name = "mola"

    def apply(self, model_args, training_args):
        base_model = AutoModelForCausalLM.from_pretrained(
            training_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
        )

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            training_args.tokenizer,
            padding_side="left",
            use_fast=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        if base_model.get_input_embeddings().weight.size(0) != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))

        # Load MoLA model
        mola_config = MolaConfig(
            lora_rank=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=training_args.lora_targets.split(","),
            mola_num_experts=training_args.mola_num_experts,
            mola_top_k=training_args.mola_top_k,
            mola_use_null_expert=training_args.mola_use_null_expert,
            mola_num_null_experts=training_args.mola_num_null_experts,
            mola_output_router_logits=training_args.mola_output_router_logits,
            mola_router_aux_loss_coef=training_args.mola_router_aux_loss_coef,
            mola_null_expert_penalty=training_args.mola_null_expert_penalty,
            mola_aux_loss_annealing=training_args.mola_aux_loss_annealing,
            mola_debug_mode=training_args.mola_debug_mode,
        )
        model = MolaModel(base_model, mola_config)

        # unfreeze embedding layers
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = True
        for param in model.get_output_embeddings().parameters():
            param.requires_grad = True

        model.print_trainable_parameters()

        #print("\n--- Verifying Trainable Parameters (MoLA) ---")
        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print(f"  - {name}")
        #print("---------------------------------------------\n")

        return model, tokenizer, {}

    def aux_loss(self, model):
        if hasattr(model, "get_aux_loss") and callable(model.get_aux_loss):
            return model.get_aux_loss()
        warnings.warn("Model has no get_aux_loss, using 0.0 tensor.")
        return torch.tensor(0.0, device=next(model.parameters()).device)
