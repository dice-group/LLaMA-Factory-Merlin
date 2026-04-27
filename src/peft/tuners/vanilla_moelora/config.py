from dataclasses import dataclass, field

from ..lora import LoraConfig
from ...utils.peft_types import PeftType


@dataclass
class VanillaMoELoraConfig(LoraConfig):
    num_experts: int = field(default=4, metadata={"help": "Number of LoRA experts per adapted linear layer."})
    top_k: int = field(default=2, metadata={"help": "Top-k experts selected per token."})
    router_aux_loss_coef: float = field(default=0.001, metadata={"help": "Load-balancing auxiliary loss weight."})
    router_init_range: float = field(default=0.02, metadata={"help": "Router weight initialization stddev."})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.VANILLA_MOELORA
