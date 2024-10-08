"""Script to build transformers"""

from model.transformer import Transformer
import torch
import torch.nn as nn

def build_transformer(default=True):
    if not default:
        raise NotImplementedError
    vocab_size = 10000
    embedding_dims = 512
    num_encoders = 6
    num_decoders = 6
    num_heads = 8
    dropout = 0.1
    expansion_dims = 2048
    sequence_length = 100
    source_embedding_layer_params = {
        "embedding_dims":embedding_dims,
        "vocab_size":vocab_size
    }
    source_positional_encoder_params = {
        "embedding_dims": embedding_dims,
        "sequence_length": sequence_length,
        "dropout": dropout
    }
    encoder_params = {
        "embedding_dims": embedding_dims,
        "num_encoders": num_decoders,
        "expansion_dims": expansion_dims,
        "source_mask": None,
        "num_heads": num_heads,
        "dropout": dropout
    }
    target_embedding_layer_params = {
        "embedding_dims":embedding_dims,
        "vocab_size":vocab_size
    }
    target_positional_encoder_params = {
        "embedding_dims": embedding_dims,
        "sequence_length": sequence_length,
        "dropout": dropout
    }
    decoder_params = {
        "num_decoders": num_decoders,
        "embedding_dims": embedding_dims,
        "expansion_dims": expansion_dims,
        "target_mask": None,
        "source_mask": None,
        "num_heads": num_heads,
        "dropout": dropout
    }
    output_layer_params = {
        "embedding_dims":embedding_dims,
        "vocab_size":vocab_size
    }


    transformer = Transformer(source_embedding_layer_params, source_positional_encoder_params, encoder_params, target_embedding_layer_params, target_positional_encoder_params, decoder_params, output_layer_params)
    for p in transformer.parameters():
        if p.dim() > 1: 
            nn.init.xavier_uniform_(p) #Xavier uniform random initiation

    return transformer