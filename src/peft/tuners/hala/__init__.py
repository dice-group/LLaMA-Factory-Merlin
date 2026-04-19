from peft.utils import register_peft_method

from .config import HalaConfig
from .layer import Embedding, HalaLoraLayer, Linear
from .model import HalaModel


__all__ = ["HalaConfig", "Embedding", "HalaLoraLayer", "Linear", "HalaModel"]

register_peft_method(name="hala", config_cls=HalaConfig, model_cls=HalaModel, prefix="lora_")
