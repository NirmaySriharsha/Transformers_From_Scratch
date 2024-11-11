"""Transformer Class with the same architecture as described in the paper Attention Is All You Need"""

import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from output_layer import OutputLayer
from embedding_layer import EmbeddingLayer
from positional_encoding import PositionalEncoder

class Transformer(nn.Module):
    """Transformer class"""

    def __init__(self, source_embedding_layer_params: dict, source_positional_encoder_params: dict, encoder_params: dict,
                 target_embedding_layer_params: dict, target_positional_encoder_params: dict, decoder_params: dict, output_layer_params: dict) -> None:
        """Abstracting params as dicts for ease of reading - in practice each of these dicts will have plenty of overlap"""
        
        super().__init__()
        self.source_embedding_layer = EmbeddingLayer(**source_embedding_layer_params)
        self.source_positional_encoder = PositionalEncoder(**source_positional_encoder_params)
        self.encoder = Encoder(**encoder_params)
        self.target_embedding_layer = EmbeddingLayer(**target_embedding_layer_params)
        self.target_positional_encoder = PositionalEncoder(**target_positional_encoder_params)
        self.decoder = Decoder(**decoder_params)
        self.output_layer = OutputLayer(**output_layer_params)



    def from_components(self, source_embedding_layer: EmbeddingLayer,
                 source_positional_encoder: PositionalEncoder, 
                 encoder: Encoder, 
                 target_embedding_layer: EmbeddingLayer,
                 target_positional_encoder: PositionalEncoder,
                 decoder: Decoder,
                 output_layer: OutputLayer) -> None:
        """In case we'd rather specify the components themselves rather than the params"""
        self.source_embedding_layer = source_embedding_layer
        self.source_positional_layer = source_positional_encoder
        self.encoder = encoder
        self.target_embedding_layer = target_embedding_layer
        self.target_positional_encoder = target_positional_encoder
        self.decoder = decoder
        self.output_layer = output_layer
        self.encoder_output = None

    def encode(self, x):
        """Embed + Positional Encode + Run through encoder

        Args:
            x (torch.tensor): Input data of shape (n_batch, seq_length, vocab_size)

        Returns:
            torch.tensor: Encoder output of shape (n_batch, seq_length, vocab_size)
        """

        #Shape of x is (n_batch, sequence_length, vocab_size)
        embedding = self.source_embedding_layer(x)
        positionally_encoded_embedding = self.source_positional_encoder(embedding)
        #shape is now (n_batch, sequence_length, embed_dims)
        encoded = self.encoder(positionally_encoded_embedding)
        #shape is now (n_batch, sequence_length, embed_dims)
        self.encoder_output = encoded
        return encoded
    
    def decode(self, x, encoder_output):
        """Embed + Positional Encode + Run through decoder"""

        #Shape of x is (n_batch, sequence_length, vocab_size)
        embedding = self.target_embedding_layer(x)
        positionally_encoded_embedding = self.target_positional_encoder(embedding)
        #shape is now (n_batch, sequence_length, embed_dims)

        #Running through decoder block requires encoder ouput for cross attention
        decoded = self.decoder(positionally_encoded_embedding, encoder_output, encoder_output)
        #shape is now (n_batch, sequence_length, embed_dims)
        return decoded
    
    def output(self, x):
        """Get final output of shape (n_batch, seq_length, vocab_size) consisting of probabilities for each token in vocabulary"""
        output = self.output_layer(x)
        return output
    
    def forward(self, x):
        """Forward pass of the transformer"""
        if self.encoder_output is None: #No need to recompute encoder output per new token generated
            self.sencoder_output = self.encode(x)
        decoder_output = self.decode(x, self.encoder_output)
        output = self.output_layer(decoder_output)
        return output
        

        
