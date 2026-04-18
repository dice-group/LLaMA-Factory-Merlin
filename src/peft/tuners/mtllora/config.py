from dataclasses import dataclass, field
from typing import Literal, Optional

from ..lora import LoraConfig
from ...utils.peft_types import PeftType


@dataclass
class MtlLoraConfig(LoraConfig):
    task_num: int = field(default=1, metadata={"help": "Number of task ids used by MTL-LoRA."})
    num_up_projections: int = field(default=3, metadata={"help": "Number of shared MTL-LoRA B/up-projection matrices."})
    temperature: float = field(default=0.1, metadata={"help": "Softmax temperature for task-specific B mixing."})
    lambda_format: Literal["full", "diagonal"] = field(
        default="full",
        metadata={"help": "Task-specific lambda transform format: full r x r matrix or diagonal vector."},
    )
    use_language_ids_as_task_ids: bool = field(
        default=True,
        metadata={"help": "Use batch language_ids as MTL-LoRA task ids."},
    )
    language_list: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Ordered language list aligned with language_ids for eval-time task injection."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.MTLLORA
        self.lambda_format = str(self.lambda_format).strip().lower()
        if self.lambda_format not in {"full", "diagonal"}:
            raise ValueError("`lambda_format` must be either 'full' or 'diagonal'.")
