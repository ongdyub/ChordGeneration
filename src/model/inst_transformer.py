import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn import Transformer
import torch.nn.functional as F
import math
import random
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]  # Broadcasting positional encodings
        return x  # Shape: (batch_size, seq_len, d_model)
    
class NormEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(NormEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x, lengths):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]  # Broadcasting positional encodings
        return x  # Shape: (batch_size, seq_len, d_model)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0  # d_model must be divisible by num_heads
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])  # Q, K, V, and output projections

    def forward(self, query, key, value, mask=None):
        # query, key, value shape: (batch_size, seq_len, d_model)
        batch_size = query.size(0)

        # Linear projections
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # After projection and reshaping: (batch_size, num_heads, seq_len, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, value)
        # attn_output shape: (batch_size, num_heads, seq_len, d_k)

        # Concatenate heads and apply final linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.linears[-1](attn_output)  # Shape: (batch_size, seq_len, d_model)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return self.linear2(F.relu(self.linear1(x)))  # Shape: (batch_size, seq_len, d_model)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, d_model)
        attn_output = self.self_attn(x, x, x, mask)  # Self-attention
        x = x + self.dropout(attn_output)  # Add & Norm
        x = self.layernorm1(x)

        ff_output = self.feed_forward(x)  # Feed forward
        x = x + self.dropout(ff_output)  # Add & Norm
        return self.layernorm2(x)  # Shape: (batch_size, seq_len, d_model)
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # x shape: (batch_size, tgt_len, d_model)
        # memory shape: (batch_size, src_len, d_model)

        attn_output = self.self_attn(x, x, x, tgt_mask)  # Self-attention
        x = x + self.dropout(attn_output)  # Add & Norm
        x = self.layernorm1(x)

        attn_output = self.cross_attn(x, memory, memory, src_mask)  # Cross-attention
        x = x + self.dropout(attn_output)  # Add & Norm
        x = self.layernorm2(x)

        ff_output = self.feed_forward(x)  # Feed forward
        x = x + self.dropout(ff_output)  # Add & Norm
        return self.layernorm3(x)  # Shape: (batch_size, tgt_len, d_model)
    
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, src_len)
        x = self.embedding(src)  # Embedding
        # x shape: (batch_size, src_len, d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x  # Shape: (batch_size, src_len, d_model)
    
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, src_mask=None, tgt_mask=None):
        # tgt shape: (batch_size, tgt_len)
        x = self.embedding(tgt)  # Embedding
        # x shape: (batch_size, tgt_len, d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x  # Shape: (batch_size, tgt_len, d_model)
    
class InstDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len, dropout=0.1):
        super(InstDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.onehot_to_dmodel = nn.Linear(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, src_mask=None, tgt_mask=None):
        # tgt shape: (batch_size, tgt_len, n_class)
        x = self.onehot_to_dmodel(tgt)  # Embedding
        # x shape: (batch_size, tgt_len, d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x  # Shape: (batch_size, tgt_len, d_model)
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, src_vocab_size, max_len, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, tgt_vocab_size, max_len, dropout)
        self.out = nn.Linear(d_model, tgt_vocab_size)
        
    def generate_src_mask(self, src):
        # src shape: (batch_size, src_len)
        return (src != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, src_len)

    def generate_tgt_mask(self, tgt):
        # tgt shape: (batch_size, tgt_len)
        tgt_len = tgt.size(1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # Padding mask: (batch_size, 1, 1, tgt_len)
        nopeak_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()  # Look-ahead mask
        tgt_mask = tgt_mask & nopeak_mask.unsqueeze(0)  # Combined mask: (batch_size, 1, tgt_len, tgt_len)
        return tgt_mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src shape: (batch_size, src_len)
        # tgt shape: (batch_size, tgt_len)
        
        src_mask = self.generate_src_mask(src)  # Generate source mask
        tgt_mask = self.generate_tgt_mask(tgt)  # Generate target mask

        memory = self.encoder(src, src_mask)  # Encoder output
        # memory shape: (batch_size, src_len, d_model)

        output = self.decoder(tgt, memory, src_mask, tgt_mask)  # Decoder output
        # output shape: (batch_size, tgt_len, d_model)

        return self.out(output)  # Final output projection, shape: (batch_size, tgt_len, tgt_vocab_size)
    
    
class InstEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, inst_size, max_len, dropout=0.1):
        super(InstEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.inst_proj = nn.Linear(d_model, inst_size)
        
    def generate_src_mask(self, src):
        # src shape: (batch_size, src_len)
        return (src != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, src_len)

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, src_len)
        x = self.embedding(src)  # Embedding
        # x shape: (batch_size, src_len, d_model)
        
        src_mask = self.generate_src_mask(src)  # Generate source mask
        
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.inst_proj(x)
        return x  # Shape: (batch_size, src_len, inst_size)
    
class InstNoPEEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, inst_size, max_len, dropout=0.1):
        super(InstNoPEEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.inst_proj = nn.Linear(d_model, inst_size)
        
    def generate_src_mask(self, src):
        # src shape: (batch_size, src_len)
        return (src != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, src_len)

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, src_len)
        x = self.embedding(src)  # Embedding
        # x shape: (batch_size, src_len, d_model)
        
        src_mask = self.generate_src_mask(src)  # Generate source mask
        
        # x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.inst_proj(x)
        return x  # Shape: (batch_size, src_len, inst_size)

class InstNormEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, inst_size, max_len, dropout=0.1):
        super(InstEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model//2)
        self.norm_pos = nn.Embedding(100, d_model//2)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.inst_proj = nn.Linear(d_model, inst_size)
        
    def generate_src_mask(self, src):
        # src shape: (batch_size, src_len)
        return (src != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, src_len)

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, src_len)
        x = self.embedding(src)  # Embedding
        norm_pos = self.embedding()
        # x shape: (batch_size, src_len, d_model)
        
        src_mask = self.generate_src_mask(src)  # Generate source mask
        
        # x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.inst_proj(x)
        return x  # Shape: (batch_size, src_len, inst_size)

class InstTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_len=5000, dropout=0.1):
        super(InstTransformer, self).__init__()
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, src_vocab_size, max_len, dropout)
        self.decoder = InstDecoder(d_model, num_heads, d_ff, num_layers, tgt_vocab_size, max_len, dropout)
        self.out = nn.Linear(d_model, tgt_vocab_size)
        # self.inst_num = 132
        
    def generate_src_mask(self, src):
        # src shape: (batch_size, src_len)
        return (src != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, src_len)

    def generate_tgt_mask(self, tgt):
        # tgt shape: (batch_size, tgt_len, n_class)
        tgt = torch.mean(tgt, dim=2)
        tgt_len = tgt.size(1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # Padding mask: (batch_size, 1, 1, tgt_len)
        nopeak_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()  # Look-ahead mask
        tgt_mask = tgt_mask & nopeak_mask.unsqueeze(0)  # Combined mask: (batch_size, 1, tgt_len, tgt_len)
        return tgt_mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src shape: (batch_size, src_len)
        # tgt shape: (batch_size, tgt_len)
        
        src_mask = self.generate_src_mask(src)  # Generate source mask
        tgt_mask = self.generate_tgt_mask(tgt)  # Generate target mask

        memory = self.encoder(src, src_mask)  # Encoder output
        # memory shape: (batch_size, src_len, d_model)

        output = self.decoder(tgt, memory, src_mask, tgt_mask)  # Decoder output
        # output shape: (batch_size, tgt_len, d_model)

        return self.out(output)  # Final output projection, shape: (batch_size, tgt_len, tgt_vocab_size)
    
    def infer(self, src, tgt, length=1024):
        output = self.forward(src, tgt)