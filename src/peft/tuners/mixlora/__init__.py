from peft.utils import register_peft_method

from .config import MixLoraConfig
from .layer import MixLoraMoeLayer
from .model import MixLoraModel

__all__ = ["MixLoraConfig", "MixLoraMoeLayer", "MixLoraModel"]

register_peft_method(name="mixlora", config_cls=MixLoraConfig, model_cls=MixLoraModel, prefix="lora_")
