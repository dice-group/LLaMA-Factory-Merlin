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

import torch

import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from peft import MovLoraConfig, TaskType, get_peft_model
from peft.tuners.movlora.layer import LinearMovLoraLayer
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict


def _build_base_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config)


def _build_mov_model():
    model = _build_base_model()
    config = MovLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=["k_proj", "v_proj", "down_proj"],
        num_experts=4,
        router_top_k=0,
        router_temperature=1.0,
        router_jitter_noise=0.0,
        router_bias=False,
        router_init_std=0.02,
        router_ignore_padding_tokens=False,
    )
    return get_peft_model(model, config)


@pytest.mark.runs_on(["cpu", "npu", "cuda"])
def test_movlora_matches_paper_target_scope_and_shapes():
    model = _build_mov_model()
    layer = model.base_model.model.model.layers[0]

    assert isinstance(layer.self_attn.k_proj, LinearMovLoraLayer)
    assert isinstance(layer.self_attn.v_proj, LinearMovLoraLayer)
    assert isinstance(layer.mlp.down_proj, LinearMovLoraLayer)
    assert not isinstance(layer.self_attn.q_proj, LinearMovLoraLayer)
    assert not isinstance(layer.self_attn.o_proj, LinearMovLoraLayer)

    k_scaling = layer.self_attn.k_proj.lora_mov_scaling["default"]
    v_scaling = layer.self_attn.v_proj.lora_mov_scaling["default"]
    ff_scaling = layer.mlp.down_proj.lora_mov_scaling["default"]

    assert tuple(k_scaling.shape) == (layer.self_attn.num_key_value_heads, 4, layer.self_attn.head_dim)
    assert tuple(v_scaling.shape) == (layer.self_attn.num_key_value_heads, 4, layer.self_attn.head_dim)
    assert tuple(ff_scaling.shape) == (4, layer.mlp.down_proj.in_features)

    assert layer.self_attn.k_proj.router_headwise["default"] is True
    assert layer.self_attn.v_proj.router_headwise["default"] is True
    assert layer.mlp.down_proj.router_headwise["default"] is False

    input_ids = torch.randint(0, model.config.vocab_size, (2, 5), device=model.device)
    outputs = model(input_ids=input_ids)
    assert outputs.logits.shape == (2, 5, model.config.vocab_size)


@pytest.mark.runs_on(["cpu", "npu", "cuda"])
def test_movlora_state_dict_round_trip():
    torch.manual_seed(42)
    model_a = _build_mov_model()
    torch.manual_seed(42)
    model_b = _build_mov_model()

    layer = model_a.base_model.model.model.layers[0]
    layer.self_attn.k_proj.lora_router["default"].weight.data.normal_()
    layer.self_attn.v_proj.lora_mov_scaling["default"].data.normal_()
    layer.mlp.down_proj.lora_router["default"].weight.data.normal_()
    layer.mlp.down_proj.lora_mov_scaling["default"].data.normal_()

    adapter_state = get_peft_model_state_dict(model_a)
    assert any(".self_attn.k_proj.lora_router.weight" in key for key in adapter_state)
    assert any(".self_attn.v_proj.lora_mov_scaling" in key for key in adapter_state)
    assert any(".mlp.down_proj.lora_mov_scaling" in key for key in adapter_state)
    assert not any(".self_attn.q_proj." in key for key in adapter_state)
    assert not any(".self_attn.o_proj." in key for key in adapter_state)

    load_result = set_peft_model_state_dict(model_b, adapter_state)
    missing_adapter = [key for key in load_result.missing_keys if "lora_router" in key or "lora_mov_scaling" in key]

    assert not load_result.unexpected_keys
    assert not missing_adapter

    input_ids = torch.randint(0, model_a.config.vocab_size, (2, 5))
    with torch.no_grad():
        logits_a = model_a(input_ids=input_ids).logits
        logits_b = model_b(input_ids=input_ids).logits

    assert torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5)
