from typing import Any, Optional

import torch
import torch.nn as nn

from ..hydralora.layer import HydraLoraLayer, Linear as HydraLinear
from .forward import forward_soft_moe


class SoftMoeLoraLayer(HydraLoraLayer):
    soft_moe_temperature: float = 1.0


class Linear(HydraLinear):

    def __init__(self, base_layer: nn.Module, adapter_name: str, **kwargs):
        super().__init__(base_layer, adapter_name, **kwargs)
        self.soft_moe_temperature = kwargs.get("soft_moe_temperature", 1.0)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        language_ids = kwargs.pop("language_ids", None)
        if language_ids is None:
            language_ids = getattr(self, "language_ids", None)
        return forward_soft_moe(self, x, *args, language_ids=language_ids, **kwargs)


class Embedding(nn.Module, SoftMoeLoraLayer):
    def __init__(self, base_layer: nn.Module, adapter_name: str, **kwargs):
        super().__init__()
        SoftMoeLoraLayer.__init__(self, base_layer)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.base_layer(x, *args, **kwargs)
