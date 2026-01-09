import torch
import torch.nn as nn
from typing import Dict
from dataclasses import dataclass
from typing import Any, Tuple

import sys
from pathlib import Path

from ml_tools.metrics.core import get_batch_metrics
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
            batch = {
                'pred': model(data['pre-event image'].to(self.device)),
                'label': labels
            }
            
            batch_metrics = get_batch_metrics(batch, {'ce': self.loss_fn}, task="segmentation")
            
        if state['get_buffers']:
            self.buffers.add(
                preds=batch['pred'],
                labels=batch['label']
            )
        
        return batch_metrics