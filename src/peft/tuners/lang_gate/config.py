from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from peft.utils import PeftType

from ..hydralora.config import HydraLoraConfig


@dataclass
class LangGateConfig(HydraLoraConfig):
    lang_gate_type: Literal["sigmoid", "softmax"] = field(
        default="sigmoid",
        metadata={"help": "Gate activation: 'sigmoid' (independent dims) or 'softmax' (competitive)."},
    )
    lang_gate_init: Literal["ones", "identity"] = field(
        default="ones",
        metadata={"help": "Gate initialization: 'ones' (fully shared) or 'identity' (separated)."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.LANG_GATE
        self.use_hydralora_experts = False
        if self.language_list is None or len(self.language_list) < 2:
            raise ValueError("LangGate requires at least 2 languages in language_list.")
        if self.lang_gate_init == "identity" and len(self.language_list) > self.r:
            raise ValueError(
                f"'identity' init requires num_languages ({len(self.language_list)}) <= rank ({self.r})."
            )
