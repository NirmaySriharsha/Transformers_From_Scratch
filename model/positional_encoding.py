"""Implement positional encoding.
The embedding layer does not retain positional information about the input sentences and thus needs
to be augmented with positional encoding. 
As per the paper we use a fixed positional encoding (i.e, we do not learn it) as follows: 
For each sentence and each position pos, the Positional Encoding PE is given by 
PE(pos, 2i) = sin(pos/10000**(2i/embedding_dims))
PE(pos, 2i+1) = cos(pos/10000**(2i/embedding_dims))
where i is the dimension from 0 to embedding_dims
We simply add the positional encoding to the original vector to augment it with positional data"""

import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    """Adds Positional Encoding to embedded sentences"""

    def __init__(self, embedding_dims: int, sequence_length: int, dropout: float) -> None:
        """Initializes and calculates positional encoding

        Args:
            embedding_dims (int): Dimensions word vectors are embedded into, usually 256 or 512
            sequence_length (int): Max Length of sentence in the case of language transformers
            dropout (float): Determines sparsity of layer
        """
         
        super().__init__()
        self.embedding_dims = embedding_dims
        self.sequence_length = sequence_length
        #A dropout layer to introduce sparsity
        self.dropout = nn.Dropout(dropout)
        
        #Note that the positional encoding is fixed and not learnt. So we calculate during initialization
        #rather than during forward pass.

        #Create a matrix positional_embedding of shape (sequence_length, embedding_dims) so that 
        #positional_embedding[pos, i] corresponds to the formula above
        positional_encoding = torch.zeros(sequence_length, embedding_dims)

        #Create a vector corresponding to the denominator inside the trig: 10000**(2i/embedding_dims)
        denominator_exponent = torch.arange(0, embedding_dims, 2).float()/embedding_dims
        denominator = torch.exp(denominator_exponent*torch.log(10000)) #We take exp(log(..)) to prevent underflow
        #Create the numerator pos in the equation inside the trig
        pos_numerator = torch.arange(0, sequence_length, dtype = torch.float).unsqueeze(1) #shape (sequence_length, 1)

        #PE(pos, 2i)
        positional_encoding[:, 0::2] = torch.sin(pos_numerator/denominator)

        #PE(pos, 2i+1)
        positional_encoding[:, 1::2] = torch.cos(pos_numerator/denominator)

        positional_encoding = positional_encoding.unsqueeze(1) #dimensions (1, seq_length, embedding_dims) for broadcasting
        
        self.positional_encoding = positional_encoding

        #We store this in the buffer for easy retrieval, since, again, we can keep it statically stored as we won't learn it
        self.register_buffer('positional_encoding', positional_encoding)


    def forward(self, x):
        """Add positional encoding

        Args:
            x (torch.ndarray): Input sentences. Shape (batch, sequence_length, embedding_dims)

        Returns:
            torch.ndarray: Positionally encoding sentences. Shape (batch, sentence_length, embedding_dim)
        """

        #Sequence_length is the maximum possible length of any input sentence, x.shape[1] is the length of this sentence
        #We encode only up till here
        positionally_encoded_x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False) 
        return positionally_encoded_x