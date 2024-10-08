"""Chain Decoder Blocks To Form A Decoder."""

import torch
import torch.nn as nn
from decoder_block import DecoderBlock
from layernorm import LayerNormalizer

class Decoder(nn.Module):
    """Decoder Block consists of a chain of decoders followed by a layer norm"""

    def __init__(self, num_decoders: int, embedding_dims: int, expansion_dims: int, target_mask: torch.tensor, source_mask: torch.tensor, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_decoders = num_decoders
        #Module list is a good way to store a sequence of modules for backprop purposes
        self.layers = nn.ModuleList([
            DecoderBlock(embedding_dims=embedding_dims, expansion_dims=expansion_dims, target_mask=target_mask, source_mask=source_mask, num_heads=num_heads, dropout=dropout)
            for _ in range(num_decoders)
        ])
        #Add a layer norm at the end
        self.layer_norm = LayerNormalizer(features=embedding_dims)
    
    def custom_layers(self, layers: nn.ModuleList):
        """In case we'd rather decide the decoder layers ourselves"""
        self.layers = layers
    
    def forward(self, x: torch.tensor, encoder_keys: torch.tensor, encoder_values: torch.tensor):
        """Forward pass of decoder

        Args:
            x (torch.tensor): input of shape (n_batch, seq_length, embedding_dims)
            encoder_keys (torch.tensor): Output of encoder for cross attention (n_batch, seq_length, embedding_dims)
            encoder_values (torch.tensor): Output of encoder for cross attention (n_batch, seq_length, embedding_dims)

        Returns:
            torch.tensor: Decoder Output
        """
        out = x
        for layer in self.layers:
            out = layer(out, encoder_keys, encoder_values)
        out = self.layer_norm(out)
        return out
