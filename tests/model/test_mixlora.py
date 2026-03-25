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

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llamafactory.train.test_utils import load_train_model
from peft import MixLoraConfig, TaskType, get_peft_model
from peft.tuners.mixlora.layer import MixLoraMoeLayer
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "mixlora",
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
    "lora_target": "q_proj,k_proj,v_proj,o_proj",
}


def _build_base_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config)


def _build_mixlora_model():
    model = _build_base_model()
    config = MixLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        moe_target_modules=["gate_proj", "up_proj", "down_proj"],
        num_experts=4,
        top_k=2,
        router_init_range=0.02,
        jitter_noise=0.0,
        router_loss=True,
        router_aux_loss_coef=1e-3,
    )
    return get_peft_model(model, config)


@pytest.mark.runs_on(["cpu", "npu", "cuda"])
def test_mixlora_train_integration():
    model = load_train_model(**TRAIN_ARGS)
    layers = model.base_model.model.model.layers

    trainable_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_names.append(name)
            assert param.dtype == torch.float32
        else:
            assert param.dtype == torch.float16

    assert any(".self_attn.q_proj.lora_A." in name for name in trainable_names)
    assert any(".mlp.lora_router." in name for name in trainable_names)
    assert any(".mlp.experts.projections.gate_proj.0.lora_A." in name for name in trainable_names)

    for layer in layers:
        assert isinstance(layer.mlp, MixLoraMoeLayer)

    input_ids = torch.randint(0, model.config.vocab_size, (2, 5), device=model.device)
    outputs = model(input_ids=input_ids)
    aux_loss = model.base_model.get_aux_loss()

    assert outputs.logits.shape == (2, 5, model.config.vocab_size)
    assert aux_loss is not None
    assert aux_loss.item() > 0


@pytest.mark.runs_on(["cpu", "npu", "cuda"])
def test_mixlora_state_dict_round_trip():
    torch.manual_seed(42)
    model_a = _build_mixlora_model()
    torch.manual_seed(42)
    model_b = _build_mixlora_model()

    mlp = model_a.base_model.model.model.layers[0].mlp
    mlp.lora_router["default"].weight.data.normal_()
    down_proj_expert = mlp.experts.projections["down_proj"][0]
    down_proj_expert.lora_A["default"].weight.data.normal_()
    down_proj_expert.lora_B["default"].weight.data.normal_()
    model_a.base_model.model.model.layers[0].self_attn.q_proj.lora_A["default"].weight.data.normal_()
    model_a.base_model.model.model.layers[0].self_attn.q_proj.lora_B["default"].weight.data.normal_()

    adapter_state = get_peft_model_state_dict(model_a)
    assert any("lora_router.weight" in key for key in adapter_state)
    assert any(".experts.projections.down_proj.0.lora_A.weight" in key for key in adapter_state)
    assert any(".self_attn.q_proj.lora_A.weight" in key for key in adapter_state)

    load_result = set_peft_model_state_dict(model_b, adapter_state)
    missing_adapter = [key for key in load_result.missing_keys if "lora_" in key or ".experts." in key]

    assert not load_result.unexpected_keys
    assert not missing_adapter

    input_ids = torch.randint(0, model_a.config.vocab_size, (2, 5))
    with torch.no_grad():
        logits_a = model_a(input_ids=input_ids).logits
        logits_b = model_b(input_ids=input_ids).logits

    assert torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5)
