from dataclasses import dataclass, field
from typing import Optional, Union

from ..ia3 import IA3Config
from ...utils.peft_types import PeftType


@dataclass
class MovLoraConfig(IA3Config):
    num_experts: int = field(default=30, metadata={"help": "Number of IA3-vector experts per adapted module."})
    feedforward_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "Module names (or regex) treated as feedforward modules; these use input-side IA3 scaling."
        },
    )
    router_top_k: int = field(default=0, metadata={"help": "Optional sparse top-k routing (<=0 keeps dense soft routing)."})
    router_temperature: float = field(default=1.0, metadata={"help": "Softmax temperature for expert routing."})
    router_jitter_noise: float = field(default=0.0, metadata={"help": "Multiplicative jitter noise amplitude applied to router inputs during training."})
    router_bias: bool = field(default=False, metadata={"help": "Whether to enable bias in router projection."})
    router_init_std: float = field(default=2e-2, metadata={"help": "Router weight initialization stddev."})
    router_ignore_padding_tokens: bool = field(default=False, metadata={"help": "Mask all-zero token states when computing routing probabilities."})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.MOVLORA
        self.feedforward_modules = (
            set(self.feedforward_modules) if isinstance(self.feedforward_modules, list) else self.feedforward_modules
        )
