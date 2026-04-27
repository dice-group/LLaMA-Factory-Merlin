import torch
from transformers import LlamaConfig, LlamaForCausalLM

from peft import TaskType, VanillaMoELoraConfig, get_peft_model
from peft.tuners.vanilla_moelora.layer import LinearVanillaMoELoraLayer
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict


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


def _build_vanilla_moelora_model():
    config = VanillaMoELoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        num_experts=4,
        top_k=2,
        router_aux_loss_coef=1e-3,
    )
    return get_peft_model(_build_base_model(), config)


def test_vanilla_moelora_forward_and_aux_loss():
    model = _build_vanilla_moelora_model()
    q_proj = model.base_model.model.model.layers[0].self_attn.q_proj

    assert isinstance(q_proj, LinearVanillaMoELoraLayer)
    assert len(q_proj.lora_A["default"]) == 4
    assert "default" in q_proj.lora_router

    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    outputs = model(input_ids=input_ids)
    aux_loss = model.base_model.get_aux_loss()

    assert outputs.logits.shape == (2, 5, model.config.vocab_size)
    assert aux_loss is not None
    assert aux_loss.item() > 0


def test_vanilla_moelora_state_dict_round_trip():
    torch.manual_seed(42)
    model_a = _build_vanilla_moelora_model()
    torch.manual_seed(42)
    model_b = _build_vanilla_moelora_model()

    q_proj = model_a.base_model.model.model.layers[0].self_attn.q_proj
    q_proj.lora_router["default"].weight.data.normal_()
    q_proj.lora_A["default"][0].weight.data.normal_()
    q_proj.lora_B["default"][0].weight.data.normal_()

    adapter_state = get_peft_model_state_dict(model_a)
    assert any("lora_router.weight" in key for key in adapter_state)
    assert any("lora_A.0.weight" in key for key in adapter_state)

    load_result = set_peft_model_state_dict(model_b, adapter_state)
    missing_adapter = [key for key in load_result.missing_keys if "lora_" in key]

    assert not load_result.unexpected_keys
    assert not missing_adapter

    input_ids = torch.randint(0, model_a.config.vocab_size, (2, 5))
    with torch.no_grad():
        logits_a = model_a(input_ids=input_ids).logits
        logits_b = model_b(input_ids=input_ids).logits

    assert torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5)
