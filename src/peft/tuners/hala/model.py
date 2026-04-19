from __future__ import annotations

from peft.tuners.tuners_utils import BaseTunerLayer

from ..hydralora.model import HydraLoraModel
from .layer import dispatch_default


class HalaModel(HydraLoraModel):
    prefix: str = "lora_"

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        current_mode = getattr(lora_config, "hala_execution_mode", "dense_expert_dense_head")
        previous_mode = getattr(lora_config, "_hala_execution_mode_runtime", None)
        lora_config._hala_execution_mode_runtime = current_mode
        try:
            super()._create_and_replace(
                lora_config,
                adapter_name,
                target,
                target_name,
                parent,
                current_key,
            )
        finally:
            if previous_mode is None:
                try:
                    delattr(lora_config, "_hala_execution_mode_runtime")
                except AttributeError:
                    pass
            else:
                lora_config._hala_execution_mode_runtime = previous_mode

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        kwargs["hala_execution_mode"] = getattr(
            lora_config, "_hala_execution_mode_runtime", getattr(lora_config, "hala_execution_mode", "dense_expert_dense_head")
        )
        dispatchers = []
        if lora_config._custom_modules:
            def dynamic_dispatch_func(target, adapter_name, lora_config, **kwargs):
                new_module = None
                if isinstance(target, BaseTunerLayer):
                    target_base_layer = target.get_base_layer()
                else:
                    target_base_layer = target
                for key, custom_cls in lora_config._custom_modules.items():
                    if isinstance(target_base_layer, key):
                        new_module = custom_cls(target, adapter_name, **kwargs)
                        break
                return new_module

            dispatchers.append(dynamic_dispatch_func)

        dispatchers.append(dispatch_default)

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:
                break
        if new_module is None:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear`, `torch.nn.Embedding`, "
                "and `Conv1D` are supported."
            )
        return new_module
