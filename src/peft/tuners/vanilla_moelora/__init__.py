from peft.utils import register_peft_method

from .config import VanillaMoELoraConfig
from .layer import LinearVanillaMoELoraLayer, VanillaMoELoraLayer
from .model import VanillaMoELoraModel

__all__ = ["VanillaMoELoraConfig", "VanillaMoELoraLayer", "LinearVanillaMoELoraLayer", "VanillaMoELoraModel"]

register_peft_method(
    name="vanilla_moelora", config_cls=VanillaMoELoraConfig, model_cls=VanillaMoELoraModel, prefix="lora_"
)
