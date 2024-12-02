import torch
import torch.nn as nn
import math

class TriangularPositionalEncoding(nn.Module):
    """
    Applies triangular positional encoding to a 4D tensor and flattens the middle dimensions.
    
    This module:
    1. Creates position-dependent sine/cosine encodings
    2. Applies a triangular mask to create asymmetric position awareness
    3. Adds the encodings to the input tensor
    4. Flattens the two middle dimensions for compatibility with sequence models
    
    Args:
        d_model (int): The dimensionality of the model/embeddings
        
    Input shape:
        - x: [batch_size, seq_len, seq_len, d_model]
        
    Output shape:
        - encoded: [batch_size, seq_len * seq_len, d_model]
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract dimensions
        batch_size, seq_len, _, d_model = x.shape
        
        # Create position indices [seq_len, 1]
        position = torch.arange(0, seq_len, device=x.device).unsqueeze(1).float()
        
        # Create frequency terms for sine/cosine functions
        # Results in frequencies that decay exponentially from 1 to 1/10000
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        # Initialize positional encoding matrix
        pe = torch.zeros(seq_len, seq_len, d_model, device=x.device)
        
        # Fill even indices with sine and odd indices with cosine
        pe[:, :, 0::2] = torch.sin(position * div_term).unsqueeze(1).expand(-1, seq_len, -1)
        pe[:, :, 1::2] = torch.cos(position * div_term).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Create and apply triangular mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        pe = pe * mask.unsqueeze(-1) + pe.transpose(0, 1) * (1 - mask).unsqueeze(-1)
        
        # Add positional encoding and reshape
        x = (x + pe.unsqueeze(0)).reshape(batch_size, seq_len * seq_len, d_model)
        
        return x

class TriangularPositionalDecoding(nn.Module):
    """
    Removes triangular positional encoding from a flattened sequence and reshapes back to 4D.
    
    This module:
    1. Reshapes the flattened sequence back to 4D
    2. Recreates the same positional encodings
    3. Subtracts them from the input tensor
    
    Args:
        d_model (int): The dimensionality of the model/embeddings
        
    Input shape:
        - x: [batch_size, seq_len * seq_len, d_model]
        
    Output shape:
        - decoded: [batch_size, seq_len, seq_len, d_model]
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract dimensions and reshape
        batch_size, flat_seq_len, d_model = x.shape
        seq_len = int(math.sqrt(flat_seq_len))
        x = x.reshape(batch_size, seq_len, seq_len, d_model)
        
        # Create position indices
        position = torch.arange(0, seq_len, device=x.device).unsqueeze(1).float()
        
        # Create frequency terms
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        # Initialize positional encoding matrix
        pe = torch.zeros(seq_len, seq_len, d_model, device=x.device)
        
        # Fill even indices with sine and odd indices with cosine
        pe[:, :, 0::2] = torch.sin(position * div_term).unsqueeze(1).expand(-1, seq_len, -1)
        pe[:, :, 1::2] = torch.cos(position * div_term).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Create and apply triangular mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        pe = pe * mask.unsqueeze(-1) + pe.transpose(0, 1) * (1 - mask).unsqueeze(-1)
        
        # Remove positional encoding
        x = x - pe.unsqueeze(0)
        
        return x
