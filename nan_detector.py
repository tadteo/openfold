import torch
import pytorch_lightning as pl

class NaNDetector(pl.Callback):
    def __init__(self):
        super().__init__()

    def _check_nan(self, name, tensor):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            return True
        return False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if self._check_nan(f"input batch {k}", v):
                    print(f"NaN in input batch at index {batch_idx}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._check_nan("loss", outputs['loss']):
            print(f"NaN in loss at batch {batch_idx}")
        
        for name, param in pl_module.named_parameters():
            if self._check_nan(f"parameter {name}", param):
                print(f"NaN in parameter {name} after batch {batch_idx}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if outputs is not None and 'loss' in outputs:
            if self._check_nan("validation loss", outputs['loss']):
                print(f"NaN in validation loss at batch {batch_idx}")
