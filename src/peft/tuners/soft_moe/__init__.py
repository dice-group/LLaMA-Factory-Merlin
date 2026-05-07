from peft.utils import register_peft_method

from .config import SoftMoeConfig
from .layer import Embedding, Linear, SoftMoeLoraLayer
from .model import SoftMoeModel


__all__ = ["SoftMoeConfig", "Embedding", "Linear", "SoftMoeLoraLayer", "SoftMoeModel"]

register_peft_method(name="soft_moe", config_cls=SoftMoeConfig, model_cls=SoftMoeModel, prefix="lora_")
