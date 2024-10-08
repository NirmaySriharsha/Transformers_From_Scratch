"""Embedding layers to receive token and return embedding"""

import torch
import torch.nn as nn
import math

class EmbeddingLayer(nn.Module):
    """Just a wrapper around nn.Embedding for pedagogical clarity"""

    def __init__(self, embedding_dims: int, vocab_size: int) -> None:
        """Initialize

        Args:
            embedding_dims (int): Dimension of word embeddings. Usually 512 or 256.
            vocab_size (int): Size of vocabulary. 
        """
        super().__init__()
        self.embedding_dims = embedding_dims
        self.vocab_size = vocab_size
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dims) #Performs the actual embedding

    def forward(self, x):
        """Return the embedding of the token

        Args:
            x (torch.tensor): Input tokens of shape (batch, vocab_size)
        
        Returns: 
            torch.tensor: Embedding of shape (batch, embedding_size)
        """
        
        embedding_prescaled = self.embedding_layer(x)
        embedding = embedding_prescaled * math.sqrt(self.embedding_dims) #In the paper they scale embedding by sqrt(embedding_dims) (page 5)
        return embedding
