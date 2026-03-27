from dataclasses import dataclass, field
from typing import Optional

from ..lora import LoraConfig
from ...utils.peft_types import PeftType


@dataclass
class HMoRaConfig(LoraConfig):
    hidden_size: Optional[int] = field(default=None, metadata={"help": "Hidden size of the base model."})
    model_type: Optional[str] = field(default=None, metadata={"help": "Base model type."})
    torch_dtype: str = field(default="float32", metadata={"help": "Computation dtype for HMoRA modules."})

    dropout: float = field(default=0.1, metadata={"help": "Dropout used in experts, routers, and task encoder."})
    num_experts: int = field(default=8, metadata={"help": "Number of LoRA experts per adapted module."})
    use_hydra_lora: bool = field(default=False, metadata={"help": "Share matrix A across experts (Hydra-LoRA+ style)."})

    top_k_routing_strategy: bool = field(default=False, metadata={"help": "Enable top-k sparse routing."})
    top_k: int = field(default=2, metadata={"help": "Top-k experts for sparse routing."})

    use_task_router: bool = field(default=False, metadata={"help": "Enable task-level router."})
    task_router_only: bool = field(default=False, metadata={"help": "Force task-router-only routing on all layers."})
    share_router_for_qkv: bool = field(default=False, metadata={"help": "Share one router for q/k/v projections inside each layer."})
    share_router_for_w_i: bool = field(default=False, metadata={"help": "Share one router for gate/up projections inside each layer."})

    num_router_mlp_layers: int = field(default=1, metadata={"help": "Router MLP depth."})
    router_hidden_dim: int = field(default=32, metadata={"help": "Hidden dimension for multi-layer routers."})
    epsilon_alpha: float = field(default=2.0, metadata={"help": "Epsilon in hierarchical alpha schedule."})
    alpha_shift: float = field(default=0.0, metadata={"help": "Shift term in hierarchical alpha schedule."})
    alpha_low_bound: float = field(default=0.0, metadata={"help": "Lower threshold for token-router-only shortcut."})
    alpha_up_bound: float = field(default=1.0, metadata={"help": "Upper threshold for task-router-only shortcut."})

    use_load_balancing_loss: bool = field(default=False, metadata={"help": "Use Switch-style load-balancing loss."})
    use_div_loss: bool = field(default=False, metadata={"help": "Use constrained GJS divergence auxiliary loss."})
    gamma_div_certain_t: float = field(default=0.0, metadata={"help": "Token-router certainty coefficient."})
    gamma_div_balance_t: float = field(default=1.0, metadata={"help": "Token-router balance coefficient."})
    gamma_div_certain_s: float = field(default=0.0, metadata={"help": "Task-router certainty coefficient."})
    gamma_div_balance_s: float = field(default=1.0, metadata={"help": "Task-router balance coefficient."})
    lambda_auxiliary: float = field(default=0.01, metadata={"help": "Auxiliary-loss weight."})
    lambda_lm: float = field(default=1.0, metadata={"help": "Language-model loss weight."})
    eta_b: float = field(default=1.0, metadata={"help": "Learning-rate multiplier for expert matrix B (Hydra-LoRA+)."})

    target_modules_lora: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Optional subset of target modules using single LoRA (no routing), e.g. o_proj,down_proj."},
    )

    use_language_ids_as_task_ids: bool = field(
        default=False,
        metadata={"help": "Use batch `language_ids` as task ids for selecting task embeddings in multilingual HMoRA."},
    )
    num_task_embeddings: int = field(
        default=1,
        metadata={"help": "Number of task-embedding rows reserved for HMoRA task ids."},
    )

    task_token: str = field(default="?", metadata={"help": "Token used to initialize the task embedding."})
    task_token_id: Optional[int] = field(default=None, metadata={"help": "Explicit task token id for task embedding init."})
    num_encoder_layer: int = field(default=1, metadata={"help": "Number of transformer layers in the task encoder."})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.HMORA
        self.torch_dtype = str(self.torch_dtype).replace("torch.", "")
        if isinstance(self.target_modules_lora, str):
            values = [item.strip() for item in self.target_modules_lora.split(",") if item.strip()]
            self.target_modules_lora = values or None
        elif self.target_modules_lora is not None:
            self.target_modules_lora = [str(item).strip() for item in self.target_modules_lora if str(item).strip()]
        self.num_task_embeddings = max(int(self.num_task_embeddings or 1), 1)
