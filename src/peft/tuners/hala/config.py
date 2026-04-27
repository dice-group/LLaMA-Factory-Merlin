from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from peft.utils import PeftType

from ..hydralora.config import HydraLoraConfig


@dataclass
class HalaConfig(HydraLoraConfig):
    hala_execution_mode: Literal[
        "dense_expert_dense_head",
        "sparse_expert_dense_head",
        "packed_sparse_expert_dense_head",
        "packed_dense_lowrank",
    ] = field(
        default="dense_expert_dense_head",
        metadata={
            "help": (
                "HALA execution mode. `dense_expert_dense_head` keeps dense compute across both stages, "
                "`sparse_expert_dense_head` keeps sparse expert dispatch with dense head mixing, and "
                "`packed_sparse_expert_dense_head` also packs per-expert B-head projections. "
                "`packed_dense_lowrank` applies HMoRA-style packed low-rank expert mixing."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.HALA
        if self.hala_execution_mode not in {
            "dense_expert_dense_head",
            "sparse_expert_dense_head",
            "packed_sparse_expert_dense_head",
            "packed_dense_lowrank",
        }:
            raise ValueError(f"Unsupported hala_execution_mode={self.hala_execution_mode!r}.")
