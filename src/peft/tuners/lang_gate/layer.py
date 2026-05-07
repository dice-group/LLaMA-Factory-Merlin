from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.metrics import record_lang_gate_metrics
from peft.tuners.tuners_utils import BaseTunerLayer


class LangGateLoraLayer(BaseTunerLayer):
    _nonlayer_adapter_attrs = frozenset({"lora_A", "lora_B", "lora_dropout", "gates"})

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.lora_dropout = nn.ModuleDict({})
        self.gates = nn.ParameterDict({})
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.language_guidance_scope = kwargs.get("language_guidance_scope", "all")
        self.language_list = kwargs.get("language_list")
        self.language_column = kwargs.get("language_column")
        self.lang_gate_type = kwargs.get("lang_gate_type", "sigmoid")
        self.lang_gate_init = kwargs.get("lang_gate_init", "ones")
        self.track_router_metrics = kwargs.get("track_router_metrics", False)
        self.num_languages = len(self.language_list) if self.language_list else 0

    def _get_gate(self, language_ids: Optional[torch.Tensor]) -> torch.Tensor:
        adapter_name = next(iter(self.gates))
        gate_param = self.gates[adapter_name]

        if language_ids is not None and torch.is_tensor(language_ids):
            gate = gate_param[language_ids]
        else:
            gate = gate_param.mean(dim=0).unsqueeze(0)

        if self.lang_gate_type == "sigmoid":
            return torch.sigmoid(gate)
        else:
            return F.softmax(gate, dim=-1)

    def _record_metrics(self, language_ids: Optional[torch.Tensor]) -> None:
        if not self.track_router_metrics:
            return
        adapter_name = next(iter(self.gates))
        gate_param = self.gates[adapter_name]
        with torch.no_grad():
            if self.lang_gate_type == "sigmoid":
                activated = torch.sigmoid(gate_param)
            else:
                activated = F.softmax(gate_param, dim=-1)

            sparsity = (activated < 0.1).float().mean().item()
            if gate_param.size(0) >= 2:
                normed = F.normalize(activated, dim=-1)
                cos_sim = (normed @ normed.T)
                mask = ~torch.eye(cos_sim.size(0), dtype=torch.bool, device=cos_sim.device)
                orthogonality = 1.0 - cos_sim[mask].mean().item()
            else:
                orthogonality = 1.0

            metrics = {
                "gate_sparsity": sparsity,
                "gate_orthogonality": orthogonality,
                "gate_mean_activation": activated.mean().item(),
            }
            record_lang_gate_metrics(metrics, weight=1.0)


class Linear(nn.Module, LangGateLoraLayer):
    def __init__(self, base_layer: nn.Module, adapter_name: str, **kwargs):
        super().__init__()
        LangGateLoraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, **kwargs)

    def update_layer(self, adapter_name: str, **kwargs):
        r = kwargs.get("r", 8)
        lora_alpha = kwargs.get("lora_alpha", 8)
        lora_dropout = kwargs.get("lora_dropout", 0.0)

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r

        in_features = self.base_layer.in_features
        out_features = self.base_layer.out_features

        if lora_dropout > 0.0:
            self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout[adapter_name] = nn.Identity()

        self.lora_A[adapter_name] = nn.Linear(in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=5**0.5)
        nn.init.zeros_(self.lora_B[adapter_name].weight)

        num_langs = self.num_languages
        if self.lang_gate_init == "ones":
            gate_data = torch.ones(num_langs, r)
        else:
            gate_data = torch.zeros(num_langs, r)
            block_size = r // num_langs
            for i in range(num_langs):
                start = i * block_size
                end = start + block_size if i < num_langs - 1 else r
                gate_data[i, start:end] = 1.0

        self.gates[adapter_name] = nn.Parameter(gate_data)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        language_ids = kwargs.pop("language_ids", None)
        if language_ids is None:
            language_ids = getattr(self, "language_ids", None)

        result = self.base_layer(x, *args, **kwargs)
        adapter_name = next(iter(self.lora_A))

        dropped = self.lora_dropout[adapter_name](x)
        h = self.lora_A[adapter_name](dropped)

        gate = self._get_gate(language_ids)
        if gate.dim() == 2 and h.dim() == 3:
            gate = gate.unsqueeze(1)
        h = h * gate

        output = self.lora_B[adapter_name](h) * self.scaling[adapter_name]
        self._record_metrics(language_ids)
        return result + output


class Embedding(nn.Module, LangGateLoraLayer):
    def __init__(self, base_layer: nn.Module, adapter_name: str, **kwargs):
        super().__init__()
        LangGateLoraLayer.__init__(self, base_layer, **kwargs)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.base_layer(x, *args, **kwargs)
