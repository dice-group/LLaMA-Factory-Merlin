from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from peft.utils import PeftType

from ..hydralora.config import HydraLoraConfig


@dataclass
class GradIsoConfig(HydraLoraConfig):
    grad_iso_num_partitions: int = field(
        default=3,
        metadata={"help": "Number of gradient-isolated LoRA partitions (one per language)."},
    )
    grad_iso_inference_mode: Literal["mean", "weighted"] = field(
        default="mean",
        metadata={"help": "How to combine partitions at inference: 'mean' averages all, 'weighted' uses language info."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.GRAD_ISO
        self.use_hydralora_experts = False
        if self.grad_iso_num_partitions < 2:
            raise ValueError("GradIso requires at least 2 partitions.")
        if self.language_list is not None and len(self.language_list) != self.grad_iso_num_partitions:
            raise ValueError(
                f"grad_iso_num_partitions ({self.grad_iso_num_partitions}) must match "
                f"language_list length ({len(self.language_list)})."
            )
