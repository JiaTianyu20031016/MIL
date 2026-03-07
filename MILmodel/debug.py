from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoConfig, PretrainedConfig, PreTrainedModel, AutoModelForSequenceClassification

class SimpleMILModel(PreTrainedModel):
    """Feeds segment batches through a pretrained backbone and averages predictions per document."""

    supported_modules = ()
    
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        *,
        backbone_name: Optional[str] = None,
    ) -> None:

        super().__init__(config=config, decision_threshold=threshold)

        backbone_kwargs = config.backbone_kwargs or {}
        self.backbone_config = AutoConfig.from_pretrained(config.backbone_name, **backbone_kwargs)
        self.backbone = AutoModelForSequenceClassification.from_pretrained(config.backbone_name, config=self.backbone_config)
        self.classifier = nn.Linear(self.backbone_config.hidden_size, 2)

    def forward(self, **input):
        return self.backbone(**input)