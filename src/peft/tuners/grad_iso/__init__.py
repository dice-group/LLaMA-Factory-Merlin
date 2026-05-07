from peft.utils import register_peft_method

from .config import GradIsoConfig
from .layer import Embedding, GradIsoLoraLayer, Linear
from .model import GradIsoModel


__all__ = ["GradIsoConfig", "Embedding", "GradIsoLoraLayer", "Linear", "GradIsoModel"]

register_peft_method(name="grad_iso", config_cls=GradIsoConfig, model_cls=GradIsoModel, prefix="lora_")
