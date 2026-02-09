from dataclasses import dataclass, field

from ..lora import LoraConfig
from ...utils.peft_types import PeftType


@dataclass
class MoELoraConfig(LoraConfig):
    expert_num: int = field(default=8, metadata={"help": "Number of LoRA experts per adapted layer."})
    task_num: int = field(default=1, metadata={"help": "Number of task ids used by the shared task gate."})
    task_embedding_dim: int = field(default=64, metadata={"help": "Task embedding dimension for the shared gate."})
    gate_top_k: int = field(
        default=0,
        metadata={"help": "Optional sparse top-k experts for the shared gate (<=0 keeps dense routing)."},
    )
    gate_temperature: float = field(default=1.0, metadata={"help": "Softmax temperature for the shared gate."})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.MOELORA
