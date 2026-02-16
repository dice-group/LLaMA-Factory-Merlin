from peft.utils import register_peft_method

from .config import MovLoraConfig
from .layer import LinearMovLoraLayer, MovLoraLayer
from .model import MovLoraModel

__all__ = ["MovLoraConfig", "MovLoraLayer", "LinearMovLoraLayer", "MovLoraModel"]

register_peft_method(name="movlora", config_cls=MovLoraConfig, model_cls=MovLoraModel, prefix="lora_")
