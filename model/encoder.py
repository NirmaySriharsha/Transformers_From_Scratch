"""Chain Encoder Blocks To Form An Encoder."""

import torch
import torch.nn as nn
from encoder_block import EncoderBlock
from layernorm import LayerNormalizer

class Encoder(nn.Module):
    """Encoders are chained Encoder Blocks with a layer normalization at the end"""

    def __init__(self, embedding_dims: int, num_encoders: int, expansion_dims: int, source_mask: torch.tensor, num_heads: int = 8, dropout: float = 0.0) -> None:
        """Create an encoder which is just a sequence of encoders followed by a layer norm

        Args:
            embedding_dims (int): Encoder parameter
            num_encoders (int): Number of encoders
            expansion_dims (int): Encoder parameter
            source_mask (torch.tensor): Encoder parameter
            num_heads (int, optional): Encoder parameter. Defaults to 8.
            dropout (float, optional): Encoder parameter. Defaults to 0.0.
        """
        super().__init__()
        self.num_encoders = num_encoders
        #Module list is a good way to store multiple modules for backprop purposes
        self.layers = nn.ModuleList([
            EncoderBlock(embedding_dims=embedding_dims, expansion_dims=expansion_dims, source_mask=source_mask, num_heads=num_heads, dropout=dropout)
            for _ in range(num_encoders)
        ])
        self.layer_normalizer = LayerNormalizer(features=embedding_dims)

    def custom_layers(self, layers: nn.ModuleList):
        """In case we'd rather decide the encoder layers ourselves"""
        self.layers = layers

    def forward(self, x):
        """Forward pass of Encoder"""
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.layer_normalizer(out)
        return out