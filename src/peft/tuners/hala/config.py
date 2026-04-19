from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from peft.utils import PeftType

from ..hydralora.config import HydraLoraConfig


@dataclass
class HalaConfig(HydraLoraConfig):
    hala_execution_mode: Literal["dense_expert_dense_head", "sparse_expert_dense_head"] = field(
        default="dense_expert_dense_head",
        metadata={
            "help": (
                "HALA execution mode. `dense_expert_dense_head` keeps dense compute across both stages, "
                "while `sparse_expert_dense_head` keeps sparse expert dispatch with dense head mixing."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.HALA
        if self.hala_execution_mode not in {"dense_expert_dense_head", "sparse_expert_dense_head"}:
            raise ValueError(f"Unsupported hala_execution_mode={self.hala_execution_mode!r}.")
