from peft.utils import register_peft_method

from .config import MixLoraConfig
from .layer import MixLoraMoeLayout
from .model import MixLoraModel

__all__ = ["MixLoraConfig", "MixLoraMoeLayout", "MixLoraModel"]

register_peft_method(name="mixlora", config_cls=MixLoraConfig, model_cls=MixLoraModel, prefix="lora_")
