from peft.utils import register_peft_method

from .config import MolaConfig
from .layer import LinearMolaLayer, MolaLayer
from .model import MolaModel

__all__ = ["MolaConfig", "MolaLayer", "LinearMolaLayer", "MolaModel"]

register_peft_method(name="mola", config_cls=MolaConfig, model_cls=MolaModel, prefix="lora_")
