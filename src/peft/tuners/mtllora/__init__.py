from peft.utils import register_peft_method

from .config import MtlLoraConfig
from .layer import LinearMtlLoraLayer, MtlLoraLayer
from .model import MtlLoraModel

__all__ = ["MtlLoraConfig", "MtlLoraLayer", "LinearMtlLoraLayer", "MtlLoraModel"]

register_peft_method(name="mtllora", config_cls=MtlLoraConfig, model_cls=MtlLoraModel, prefix="lora_")
