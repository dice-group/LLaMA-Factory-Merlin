"""
AdaMoLE Initialization
"""
from peft.utils import register_peft_method

from .config import AdaMoleConfig
from .layer import AdaMoleLayer, LinearAdaMoleLayer
from .model import AdaMoleModel

__all__ = ["AdaMoleConfig", "AdaMoleLayer", "LinearAdaMoleLayer", "AdaMoleModel"]

register_peft_method(name="adamole", config_cls=AdaMoleConfig, model_cls=AdaMoleModel, prefix="lora_")


def __getattr__(name):
    raise AttributeError(f"Module {__name__} has no attribute {name}.")
