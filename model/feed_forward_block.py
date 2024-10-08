"""Feedforward block that follows a Multi Head Attention Block (and Add&Norm)
In the paper, these blocks consist of a linear expansion, a RELU and a contraction back to the original size.
Theoretically I suppose it could be many configurations but we'll assume this for our use case."""

import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    """Feed Forward Block to be placed after Attention Heads
    We will go with a default two layer NN with a RELU in between. 
    #TODO: Add customization capability down the line"""

    def __init__(self, embedding_dims: int, expansion_dims: int, dropout: float) -> None:
        """Initialize the feed forward block

        Args:
            embedding_dims (int): Embedded word dimensions
            expansion_dims (int): Size of middle layer - typically twice as big as embedding_dims
            dropout (float): Dropout parameter
        """
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dims, expansion_dims)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(expansion_dims, embedding_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass of feed forward block

        Args:
            x (torch.tensor): Output of attention block. Size of (n_batch, sequence_length, embedding_dims)

        Returns:
            torch.tensor: Forward output of size (n_batch, sequence_length, embedding_dims)
        """
        out = self.linear_1(x) #out now has dims (n_batch, sequence_length, expansion_dims)
        out = self.relu(out)
        out = self.dropout(out) #Literature varies on where dropout occurs. I don't think it makes much of a difference and am putting it here. 
        out = self.linear_2(out) #out now has dims (n_batch, sequence_length, embedding_dims)

        return out
