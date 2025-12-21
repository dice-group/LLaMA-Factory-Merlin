from peft.utils import register_peft_method

from .config import MoeConfig
from .model import MoeModel

__all__ = ["MoeConfig", "MoeModel"]

register_peft_method(name="moelpr", config_cls=MoeConfig, model_cls=MoeModel, prefix="lora_")
