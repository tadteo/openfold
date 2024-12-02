import torch
import pytorch_lightning as pl

class NaNDetector(pl.Callback):
    def __init__(self):
        super().__init__()
        self.last_valid_state = {}
        
    def _check_nan_and_log(self, name, tensor, batch_idx):
        if torch.isnan(tensor).any():
            nan_indices = torch.where(torch.isnan(tensor))
            print(f"\nNaN detected in {name} at batch {batch_idx}")
            print(f"NaN positions: {nan_indices}")
            
            if name in self.last_valid_state:
                last_valid = self.last_valid_state[name]
                print(f"Last valid value range: [{last_valid.min():.4f}, {last_valid.max():.4f}]")
            
            return True
        else:
            self.last_valid_state[name] = tensor.detach().clone()
            return False
            
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                self._check_nan_and_log(f"input/{k}", v, batch_idx)
                
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Track loss
        if isinstance(outputs, dict) and 'loss' in outputs:
            self._check_nan_and_log("loss", outputs['loss'], batch_idx)
            
        # Track gradients
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                self._check_nan_and_log(f"grad/{name}", param.grad, batch_idx)

    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
    #     if outputs is not None and 'loss' in outputs:
    #         if self._check_nan("validation loss", outputs['loss']):
    #             print(f"NaN in validation loss at batch {batch_idx}")
