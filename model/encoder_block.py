"""Chain Together Layers to Form an Encoder Block."""

import torch
import torch.nn as nn
from feed_forward_block import FeedForwardBlock
from multihead_attention_block import MultiHeadAttention
from layernorm import LayerNormalizer

class EncoderBlock(nn.Module):
    """Encoder Block Is Composed of
    MHA -> Layer Norm (Plus Skip Conn) -> FeedForward -> LayerNorm (Plus Skip Conn)"""

    def __init__(self, embedding_dims: int, expansion_dims: int, source_mask: torch.tensor, num_heads: int = 8, dropout: float = 0.0) -> None:
        """Initialize Encoder Block

        Args:
            embedding_dims (int): Embedding dimension of words
            expansion_dims (int): Middle layer size of Feed Forward Block
            source_mask (torch.tensor): Source Mask for Encoder Block to prevent padding, etc to be involved in the attention computation
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            dropout (float, optional): Dropout factor. Defaults to 0.0.
        """
        super().__init__()
        self.dropout = dropout
        self.source_mask = source_mask
        self.multi_head_attention_block = MultiHeadAttention(embedding_dims=embedding_dims, num_heads=num_heads, dropout=dropout, mask = source_mask)
        self.layer_norm_1 = LayerNormalizer(features=embedding_dims)
        self.feed_foward_block = FeedForwardBlock(embedding_dims=embedding_dims, expansion_dims=expansion_dims, dropout=dropout)
        self.layer_norm_2 = LayerNormalizer(features = embedding_dims)
    
    def forward(self, x):
        """Forward pass of encoder block
        Takes input -> splits into key, query, value by duplication -> passes through multi head self attention
        -> pass through layer normalization -> add skip connection of original input to get new output, save this as new residual
        -> pass throug feed forward layer -> pass through new layer normalization -> add skip of residual

        Args:
            x (torch.tensor): Input (positional encoded sentence embeddings that may have passed through encoder blocks already)
            shape is (n_batch, seq_length, embedding_dims)

        Returns:
            torch.tensor: output of encoder block of same shape (n_batch, seq_length, embedding_dims)
        """


        #We keep the residual for skip connection later
        residual = x

        #The attention in an encoder is SELF attention, so the queries, keys and values are all from the same source. 
        # Moreover, there is no masking for encoding since the entirety of the input is known.
        out = self.multi_head_attention_block(x, x, x)

        # Pass through the layer norm
        out = self.layer_norm_1(out)
        out += residual #Add the skip here

        residual = out #This becomes the new residual for the next skip

        #Pass through the next feed forward block
        out = self.feed_foward_block(out)
        #Pass through the next layer norm
        out = self.layer_norm_2(out)
        #Add the skip residual
        out += residual
        #And we're done
        return out
