from peft.utils import register_peft_method

from .config import MoELoraConfig
from .layer import LinearMoELoraLayer, MoELoraLayer
from .model import MoELoraModel

__all__ = ["MoELoraConfig", "MoELoraLayer", "LinearMoELoraLayer", "MoELoraModel"]

register_peft_method(name="moelora", config_cls=MoELoraConfig, model_cls=MoELoraModel, prefix="lora_")
