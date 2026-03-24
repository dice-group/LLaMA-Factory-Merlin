from dataclasses import dataclass, field
from typing import Optional, Union

from ..lora import LoraConfig
from ...utils.peft_types import PeftType


@dataclass
class MixLoraConfig(LoraConfig):
    """MixLoraConfig
    Note:
    - they handle FFN and attention differently: FFN modules are replaced with a Mixture-of-Experts (MoE) architecture where each expert is a LoRA adapter, while attention modules receive a single shared LoRA adapter.
    - jitter and init range config were copied from the original MixLoRA codebase, but are not explicitly mentioned in the paper
    """

    num_experts: int = field(default=8, metadata={"help": "Number of FFN LoRA experts per MixLoRA block."})
    top_k: int = field(default=2, metadata={"help": "Top-k FFN experts selected per token."})
    routing_strategy: str = field(
        default="mixlora",
        metadata={"help": "Routing strategy name for MixLoRA."},
    )
    router_init_range: float = field(default=2e-2, metadata={"help": "Router weight initialization stddev."})
    jitter_noise: float = field(default=0.0, metadata={"help": "Multiplicative jitter noise on router inputs."})
    router_loss: bool = field(default=True, metadata={"help": "Whether the router should contribute an auxiliary loss."})
    router_aux_loss_coef: float = field(
        default=1e-3, metadata={"help": "Auxiliary router load-balancing loss coefficient."}
    )
    act_fn: Optional[str] = field(
        default=None,
        metadata={"help": "Optional FFN activation override for MixLoRA MoE blocks. Unset uses the base MLP activation."},
    )
    moe_target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "MLP projection names used by the MixLoRA expert mechanism."},
    )
    expert_lora_r: Optional[int] = field(
        default=None, metadata={"help": "Optional expert-specific LoRA rank override for FFN experts."}
    )
    expert_lora_alpha: Optional[int] = field(
        default=None, metadata={"help": "Optional expert-specific LoRA alpha override for FFN experts."}
    )
    expert_lora_dropout: Optional[float] = field(
        default=None, metadata={"help": "Optional expert-specific LoRA dropout override for FFN experts."}
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.MIXLORA
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        self.moe_target_modules = (
            set(self.moe_target_modules) if isinstance(self.moe_target_modules, list) else self.moe_target_modules
        )
