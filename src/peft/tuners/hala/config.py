from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from peft.utils import PeftType

from ..hydralora.config import HydraLoraConfig


@dataclass
class HalaConfig(HydraLoraConfig):
    hala_execution_mode: Literal[
        "grouped_sparse_expert_dense_head",
    ] = field(
        default="grouped_sparse_expert_dense_head",
        metadata={
            "help": (
                "HALA exploration mode. The tuner only supports sparse expert top-1 routing followed by "
                "sparse head top-1 routing inside the selected expert."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.HALA
        if self.hala_execution_mode != "grouped_sparse_expert_dense_head":
            raise ValueError(f"Unsupported hala_execution_mode={self.hala_execution_mode!r}.")
        if not self.use_hydralora_experts:
            raise ValueError("HALA exploration branch requires use_hydralora_experts=True.")
        if self.top_k != 1:
            raise ValueError("HALA exploration branch requires top_k=1.")
        if self.head_top_k != 1:
            raise ValueError("HALA exploration branch requires head_top_k=1.")
        if self.language_guidance_scope != "all":
            raise ValueError("HALA requires language_guidance_scope='all'.")
        if self.language_prior_weight <= 0:
            raise ValueError("HALA requires language_prior_weight > 0 for LPR supervision.")
