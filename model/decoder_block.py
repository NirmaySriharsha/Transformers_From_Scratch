"""Chain Together Layers to Form Decoder Block."""

import torch
import torch.nn as nn
from feed_forward_block import FeedForwardBlock
from multihead_attention_block import MultiHeadAttention
from layernorm import LayerNormalizer

class DecoderBlock(nn.Module):
    """Decoder Block consists of MASKED Multi Head Attention -> Layer Norm (plus Skip Connection)
    -> Encoder/Decoder Cross Attention -> Layer Norm (plus Skip Connection) 
    -> Feed Forward Layer -> Layer Norm (plus Skip Connection)"""

    def __init__(self, embedding_dims: int, expansion_dims: int, target_mask: torch.tensor, source_mask: torch.tensor, num_heads: int = 8, dropout: float = 0.0) -> None:
        """Initialize Decoder

        Args:
            embedding_dims (int): Dimension of word embeddings
            expansion_dims (int): Dimension of middle layer of feed forward block
            target_mask (torch.tensor): Mask for self attention block. Just a lower triangular mask to prevent past tokens from peeking at future tokens.
            source_mask (torch.tensor): Mask for the encoder output - to prevent attention being calculated on paddings to input
            num_heads (int, optional): Number of attention heads, gonna assume the same across the various MHA blocks. Defaults to 8.
            dropout (float, optional): Dropout factor. Defaults to 0.0.
        """
        super().__init__()
        self.dropout = dropout
        self.target_mask = target_mask
        self.source_mask = source_mask
        self.masked_self_attention_block = MultiHeadAttention(embedding_dims=embedding_dims, num_heads=num_heads, dropout=dropout, mask=target_mask)
        self.layer_norm_1 = LayerNormalizer(features=embedding_dims)
        self.cross_attention_block = MultiHeadAttention(embedding_dims=embedding_dims, num_heads=num_heads, dropout=dropout, mask=source_mask)
        self.layer_norm_2 = LayerNormalizer(features=embedding_dims)
        self.feed_forward_block = FeedForwardBlock(embedding_dims=embedding_dims, expansion_dims=expansion_dims, dropout=dropout)
        self.layer_norm_3 = LayerNormalizer(features=embedding_dims)

    def forward(self, x, encoder_keys, encoder_values): 
        """Forward pass of a decoder block

        Args:
            x (torch.tensor): Forward pass input of shape (n_batch, seq_length, embedding_dims)
            encoder_keys (torch.tensor): Keys from encoder output (n_batch, seq_length, embedding_dims)
            encoder_values (torch.tensor): Values from encoder output (n_batch, seq_length, embedding_dims)

        Returns:
            torch.tensor: Decoder output of shape (n_batch, seq_length, embedding_dims)
        """
        #store the residual for the upcoming skip connection
        residual = x
        #pass through masked self attention
        out = self.masked_self_attention_block(x, x, x)
        #pass through layer norm 
        out = self.layer_norm_1(out)
        #add skip connection
        out += residual
        #new residual
        residual = out
        #pass through cross attention block. Note that the queries come from the decoder block but the encoder gives you the keys and values
        #this is basically where we relate the prompt to what's being generated
        out = self.cross_attention_block(out, encoder_keys, encoder_values)
        #pass through layer norm
        out = self.layer_norm_2(out)
        #add residual
        out += residual
        #new residual
        residual = out
        #pass through feed forward
        out = self.feed_forward_block(out)
        #add residual
        out += residual
        #and we're done
        return out

