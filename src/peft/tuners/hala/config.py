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
                "HALA exploration mode. The tuner supports sparse expert top-1/top-2 routing followed by "
                "sparse or dense head routing inside the selected expert."
            )
        },
    )
    hala_shared_residual: bool = field(
        default=False,
        metadata={
            "help": (
                "Add one shared LoRA residual branch before the language-routed HALA expert delta. "
                "This preserves a shared transfer path while keeping the existing routed specialization branch."
            )
        },
    )
    hala_gated_shared_capacity: bool = field(
        default=False,
        metadata={
            "help": (
                "Add a default-disabled gated shared low-rank HALA path. The routed expert branch is unchanged; "
                "the shared path is modulated by an input-conditioned scalar gate."
            )
        },
    )
    hala_gated_shared_init_bias: float = field(
        default=-4.0,
        metadata={"help": "Initial bias for the HALA gated shared-capacity scalar gate; negative starts conservative."},
    )
    hala_balance_loss_coef: float = field(
        default=0.0,
        metadata={"help": "Weight for the HALA expert-router balance auxiliary loss (0 disables)."},
    )
    hala_balance_loss_kind: Literal["none", "uniform_importance", "target_distribution_importance"] = field(
        default="none",
        metadata={"help": "HALA balance objective kind."},
    )
    hala_balance_target: Literal["expert"] = field(
        default="expert",
        metadata={"help": "HALA balance target; expert router only for the initial implementation."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.HALA
        if self.hala_execution_mode != "grouped_sparse_expert_dense_head":
            raise ValueError(f"Unsupported hala_execution_mode={self.hala_execution_mode!r}.")
        if not self.use_hydralora_experts:
            raise ValueError("HALA exploration branch requires use_hydralora_experts=True.")
        if self.top_k not in (1, 2):
            raise ValueError("HALA exploration branch requires top_k=1 or top_k=2.")
        if self.head_top_k is None or self.head_top_k not in (0, 1):
            raise ValueError("HALA exploration branch requires head_top_k=1 for sparse heads or 0 for dense heads.")
        if self.hala_balance_loss_coef < 0:
            raise ValueError("HALA balance loss coefficient must be non-negative.")
        if self.hala_balance_loss_coef > 0 and self.hala_balance_loss_kind == "none":
            raise ValueError("HALA balance loss kind must not be 'none' when coefficient is positive.")
        if self.hala_balance_loss_kind != "none" and self.hala_balance_target != "expert":
            raise ValueError("HALA balance objective currently supports expert target only.")
        if self.language_guidance_scope not in {"all", "expert_only"}:
            raise ValueError("HALA requires language_guidance_scope='all' or 'expert_only'.")
        if self.language_prior_weight <= 0 and self.language_router_mode != "hard":
            raise ValueError("HALA requires language_prior_weight > 0 unless language_router_mode='hard'.")
