"""The star of the show - Multi Headed Attention"""

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """Multi Headed Attention Block"""

    def __init__(self, embedding_dims: int, num_heads: int, dropout: float):
        """Initialize MHA"""
        super().__init__()
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        self.dropout = dropout

        assert self.head_dims * self.num_heads == self.embedding_dims, "Number of heads must be a divisor of embedding_dims"

        #Given embedding_dims and num_heads, the dims of each head is
        self.head_dims = embedding_dims // num_heads

        """Common point of misconception here:
        Technically we need num_heads number of projection matrices W_Q, W_K, W_V to project the queries, keys and values
        to different head_dims subspaces. That means each W_Q, W_K, W_V are of size (embedding_dims, head_dims). 
        For computational simplicity we concatenate these into one big W_Q, W_K, W_v of size (embedding_dims, embedding_dims)
        and then slice the outputs downstream along head_dims."""

        self.W_queries = nn.Linear(embedding_dims, embedding_dims)
        self.W_keys = nn.Linear(embedding_dims, embedding_dims) 
        self.W_values = nn.Linear(embedding_dims, embedding_dims) #Note that queries+keys can have d_k and values can have d_v dims instead
        #Where d_k != d_v is possible. But for simplicity we assume they're all equal and d_k = d_v = head_dims
        ##Note: We could add bias to these matrices but eh

        #We pass through another linear layer after attention scores are computed (and concatenated)
        self.W_out = nn.Linear(embedding_dims, embedding_dims) 

    @staticmethod #In case I want to see attention scores down the line
    def attention_score(self, queries, keys, values, dropout=False):
        """Calculates attention given queries, keys, values. 
        Calculation occurs parallel across all heads simultaneously - i.e, the main strength of transformers.

        Args:
            queries (torch.ndarray): Queries per head, shape (n_batch, num_heads, seq_length, head_dims)
            keys (torch.ndarray): Keys per head, shape (n_batch, num_heads, seq_length, head_dims)
            values (torch.ndarray): Values per head, shape (n_batch, num_heads, seq_length, head_dims)

        Returns:
            torch.ndarray: Attention per Q, K, V triad per head of shape (n_batch, num_heads, seq_length, head_dims)
        """

        #As per the formula we do QK^T but since the shapes are actually (n_batch, num_heads, seq_length, head_dims)
        #we only want to transpose the last two axes of K
        dot_product_attention = queries@keys.transpose(-2, -1)
        #scale by sqrt(head_dims) for numerical stability
        head_dims = queries.shape[-1] #Since I made this a static method I'll avoid invoking self.__ to get values
        dot_product_attention /= math.sqrt(head_dims)
        #softmax it across the head_dims space
        dot_product_attention = dot_product_attention.softmax(dim = -1)
        #shape is now (n_batch, num_heads, seq_length, seq_length) where the last two entries gives you the attention score
        #ie, for i, j <seq_length, dot_product_attention[a, :, i, j] gives you the attention between token i and j per head for some sentence a

        if dropout:
            dot_product_attention = self.dropout(dot_product_attention)

        #Multiply by values
        attention = dot_product_attention@values
        #shape is back to (n_batch, num_heads, seq_length,  head_dims) (project it back into the head_dims space)
        return attention, dot_product_attention #we return dot_product_attention as well since it's a useful thing to be able to examine

    def forward(self, queries, keys, values):
        """Forward pass!

        Args:
            queries (torch.ndarray): Queries of shape (n_batch, seq_length, embedding_dims) 
            keys (torch.ndarray): Keys of shape (n_batch, seq_length, embedding_dims) 
            values (torch.ndarray): Values of shape (n_batch, seq_length, embedding_dims) 

        Returns:
            torch.ndarray: Output of attention block of shape (n_batch, seq_length, embedding_dims)
        """
        #Pass through transformation matrix
        queries = self.W_queries(queries)
        keys = self.W_keys(keys)
        values = self.W_values(keys)
        #Each now has the shape (n_batch, seq_length, embedding_dims)
        #But we need to have shape (n_batch, seq_length, num_heads, head_dims) or better yet
        #(n_batch, head_dims, seq_length, num_heads) (for conceptual clarity as we operate on the whole sequence for each head)
        queries = self._reshape_qkv(queries)
        keys = self._reshape_qkv(keys)
        values = self._reshape_qkv(values)

        #calculate attention
        attention = self.attention_score(queries, keys, values)
        #shape of (n_batch, num_head, seq_length, head_dims)
        #Concatenate it to get shape (n_batch, seq_length, embedding_dims) once again. 
        #first shuffle back num_head and seq_length -> new shape is (n_batch, seq_length, num_head, head_dims)
        attention = attention.transpose(1, 2)

        #concatenate the last two axes
        attention = attention.contiguous().view(attention.shape[0], -1, self.num_heads * self.head_dims)
        #contiguous() is a bit weird but basically a pytorch quirk for when we reshape tensors in place. 

        #pass it through the last linear layer
        out = self.W_out(attention)
        #final shape is (n_batch, seq_length, embedding_dims)
        return out


    def _reshape_qkv(self, input):
        """Reshapes vectors (intended to be queries, keys and values) into shapes compatible for each head

        Args:
            input (torch.ndarray): A key, query or value of shape (n_batch, sequence_length, embedding_dims) concatenated num_heads times
            across each head_dims subspace

        Returns:
            torch.ndarray: key, query, or value split across each head_dims subspace. 
        """
        #Input is of shape (n_batch, sequence_length, embedding_dims). We want (n_batch, sequence_length, num_heads, head_dims)
        input = input.view(input.shape[0], input.shape[1], self.num_heads, self.head_dims)

        #input is of shape (n_batch, sequence_length, num_heads, head_dims)
        #for simplicity we would like (n_batch, num_heads, sequence_length, head_dims)
        #since each head will operate on the whole of the sequence and leave the num_heads axis alone
        input = input.transpose(1, 2)
        return input

