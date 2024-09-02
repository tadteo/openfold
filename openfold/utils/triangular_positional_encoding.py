import torch
import torch.nn as nn
import math

class TriangularPositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        
        pe = torch.zeros(seq_len, seq_len, self.d_model, device=x.device)
        pe[:, :, 0::2] = torch.sin(position * div_term).unsqueeze(1).expand(-1, seq_len, -1)
        pe[:, :, 1::2] = torch.cos(position * div_term).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Apply triangular mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        pe = pe * mask.unsqueeze(-1) + pe.transpose(0, 1) * (1 - mask).unsqueeze(-1)
        
        return x + pe.unsqueeze(0)
