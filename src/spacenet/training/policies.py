import torch
import torch.nn as nn
from typing import Dict
from dataclasses import dataclass
from typing import Any, Tuple

import sys
from pathlib import Path

from ml_tools.training.metrics import get_correct
from ml_tools.utils.buffers import EpochLogitBuffer

class TrainingPolicy:
    def compute_loss(self, data, model) -> Dict[str, torch.Tensor]: 
        ...

@dataclass
class StandardPolicy(TrainingPolicy):
    loss_fn: nn.Module
    buffers: EpochLogitBuffer
    device: torch.device
    amp_dtype: torch.dtype
    use_amp: bool = False
    
    def compute_batch_metrics(self, *, data, model, state=None):
        #get labels and masks
        labels = data['labels'].to(self.device)
        #prepare the batch
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,  # fp16 or bf16
            enabled=self.use_amp
        ):
            preds = model(data['pre-event image'].to(self.device))
            loss = self.loss_fn(preds, labels)
            
        if state['get_buffers']:
            self.buffers.add(
                logit_diffs=preds[:, 1] - preds[:, 0],
                labels=labels
            )
        batch_metrics = {}      
        batch_metrics['batch_size'] = labels.size(0)
        batch_metrics['correct'] = get_correct(preds, labels)
        batch_metrics['loss'] = loss

        return batch_metrics