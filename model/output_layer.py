"""Output layer that map decode output into vocabulary probabilities"""

import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    """I'm combining the last two layers in the paper - a linear layer and a softmax to get probability for each word/token in the vocab"""

    def __init__(self, embedding_dims: int, vocab_size: int) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(embedding_dims, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim = -1) #We do log softmax for numerical stability
    
    def forward(self, x):

        #x is of shape (n_batch, seq_length, embedding_dims)
        out = self.linear_layer(x)
        out = self.log_softmax(out)
        #out is of shape (n_batch, seq_length, vocab_size) max([:, -1, :]) gives you the most likely next token for each sentence in each batch
        return out 