from peft.utils import register_peft_method

from .config import HMoRaConfig
from .layer import HMoRaLayer, LinearHMoRaLayer
from .model import HMoRaModel

__all__ = ["HMoRaConfig", "HMoRaLayer", "LinearHMoRaLayer", "HMoRaModel"]

register_peft_method(name="hmora", config_cls=HMoRaConfig, model_cls=HMoRaModel, prefix="lora_")
