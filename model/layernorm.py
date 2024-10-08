"""Layer Normalization Module.
I implement this for completeness but won't actually use it as pytorch has an implementation already.
"""

import torch
import torch.nn as nn

class LayerNormalizer(nn.Module):
    """Apply Layer Normalization over each mini batch of inputs
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html"""

    def __init__(self, features: int, eps: float=10**-6) -> None:
        """Create Layer Normalizer. Beta is the intercept and Gamma is the scaling factor. 

        Args:
            features (int): Number of features per element in the minibatch, will be embedding_dims for most of the remainder. 
            eps (float, optional): Padding factor to prevent division by 0 erorrs/overflow. Defaults to 10**-6.
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features)) #nn.Parameter sets it to be learnable
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """Apply Layer Normalization. By default we do it across the last dimension but the pytorch implementation 
        makes this customizable which is why we we'll defer to that in the rest of the code.

        Args:
            x (torch.ndarray): minibatch layer activations being normalized.
            Shape (batch, sentence_length, features)

        Returns:
            torch.ndarray: Layer normalized output of shape (batch, sentence_length, 1)
        """

        mean = x.mean(dim = -1, keepdim = True) #keepdim doesn't remove that dimension (but makes it 1 deep obviously)
        std = x.std(dim = -1, keepdim = True) #We keepdim to help broadcasting downstream

        return (self.gamma * (x - mean) / (std + self.eps)) + self.beta
    