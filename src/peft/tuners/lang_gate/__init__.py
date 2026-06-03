from peft.utils import register_peft_method

from .config import LangGateConfig
from .layer import Embedding, LangGateLoraLayer, Linear
from .model import LangGateModel


__all__ = ["LangGateConfig", "Embedding", "LangGateLoraLayer", "Linear", "LangGateModel"]

register_peft_method(name="lang_gate", config_cls=LangGateConfig, model_cls=LangGateModel, prefix="lora_")
