from __future__ import annotations

from dataclasses import dataclass, field

from peft.utils import PeftType

from ..hydralora.config import HydraLoraConfig


@dataclass
class SoftMoeConfig(HydraLoraConfig):
    soft_moe_temperature: float = field(
        default=1.0,
        metadata={"help": "Softmax temperature for expert mixing weights. Higher = more uniform."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.SOFT_MOE
        if not self.use_hydralora_experts:
            raise ValueError("SoftMoE requires use_hydralora_experts=True.")
        if self.num_experts < 2:
            raise ValueError("SoftMoE requires at least 2 experts.")
        if self.soft_moe_temperature <= 0:
            raise ValueError("soft_moe_temperature must be positive.")
