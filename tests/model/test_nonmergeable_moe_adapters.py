from types import SimpleNamespace

import pytest
import torch

from llamafactory.model import adapter as adapter_module
from peft.tuners.adamoe_1.layer import LinearMolaLayer


def _model_args(adapters: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        adapter_name_or_path=adapters,
        adapter_folder=None,
        offload_folder=None,
        cache_dir=None,
        model_revision="main",
        hf_hub_token=None,
        use_unsloth=False,
        use_kt=False,
    )


def _finetuning_args() -> SimpleNamespace:
    return SimpleNamespace(
        use_dora=False,
        use_rslora=False,
        pissa_init=False,
        create_new_adapter=False,
    )


@pytest.mark.parametrize(
    "setup",
    [adapter_module._setup_moelora_tuning, adapter_module._setup_mola_tuning],
)
def test_routed_moe_adapter_is_resumed_without_merge(monkeypatch, setup) -> None:
    base_model = SimpleNamespace()
    loaded_model = SimpleNamespace()
    calls = []

    def fake_from_pretrained(model, adapter, **kwargs):
        calls.append((model, adapter, kwargs))
        return loaded_model

    monkeypatch.setattr(adapter_module, "is_deepspeed_zero3_enabled", lambda: False)
    monkeypatch.setattr(adapter_module.PeftModel, "from_pretrained", fake_from_pretrained)

    result = setup(
        SimpleNamespace(),
        base_model,
        _model_args(["checkpoint_adapter"]),
        _finetuning_args(),
        is_trainable=False,
        cast_trainable_params_to_fp32=False,
    )

    assert result is loaded_model
    assert len(calls) == 1
    assert calls[0][0] is base_model
    assert calls[0][1] == "checkpoint_adapter"
    assert calls[0][2]["is_trainable"] is False


@pytest.mark.parametrize(
    "setup",
    [adapter_module._setup_moelora_tuning, adapter_module._setup_mola_tuning],
)
def test_routed_moe_adapter_rejects_multiple_checkpoints(setup) -> None:
    with pytest.raises(ValueError, match="non-mergeable"):
        setup(
            SimpleNamespace(),
            SimpleNamespace(),
            _model_args(["first", "second"]),
            _finetuning_args(),
            is_trainable=False,
            cast_trainable_params_to_fp32=False,
        )


def test_mola_moe_expert_paths_are_aliases_of_saved_lora_modules() -> None:
    layer = LinearMolaLayer(
        base_layer=torch.nn.Linear(4, 3, bias=False),
        adapter_name="default",
        lora_rank=2,
        lora_alpha=2,
        num_experts=2,
        num_null_experts=1,
        top_k=1,
    )

    assert layer.moe_layer["default"].experts[0].lora_A is layer.lora_A["default"][0]
    assert layer.moe_layer["default"].experts[0].lora_B is layer.lora_B["default"][0]

    expected_a = torch.full_like(layer.lora_A["default"][0].weight, 0.25)
    expected_b = torch.full_like(layer.lora_B["default"][0].weight, -0.5)
    missing, unexpected = layer.load_state_dict(
        {
            "lora_A.default.0.weight": expected_a,
            "lora_B.default.0.weight": expected_b,
        },
        strict=False,
    )

    assert not unexpected
    assert "moe_layer.default.experts.0.lora_A.weight" in missing
    assert "moe_layer.default.experts.0.lora_B.weight" in missing
    torch.testing.assert_close(layer.moe_layer["default"].experts[0].lora_A.weight, expected_a)
    torch.testing.assert_close(layer.moe_layer["default"].experts[0].lora_B.weight, expected_b)
