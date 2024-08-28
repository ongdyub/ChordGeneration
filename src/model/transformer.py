import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import ASTConfig, ASTModel, GPT2Config, GPT2Model, AutoModelForCausalLM, GPT2LMHeadModel, BartConfig
from einops_exts import rearrange_with_anon_dims
from einops import rearrange, reduce, repeat
from torch.nn import Transformer

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000, device='cpu'):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model, requires_grad=False)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         self.pe[:, 0::2] = torch.sin(position * div_term)
#         self.pe[:, 1::2] = torch.cos(position * div_term)
#         self.pe = self.pe.unsqueeze(0)  # shape: (1, max_len, d_model)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # x: (batch_size, seq_len, d_model)
#         x = x + self.pe[:, :x.size(1), :].to(device)
#         # x: (batch_size, seq_len, d_model)
#         return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.device = device
        
        # Create positional encoding matrix
        self.register_buffer('pe', self._create_positional_encoding(max_len, d_model))

    def _create_positional_encoding(self, max_len, d_model):
        # Create a tensor to hold the positional encodings
        pe = torch.zeros(max_len, d_model, device=self.device)
        
        # Compute positional encodings
        position = torch.arange(0, max_len, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=self.device) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        return pe

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()
        
        # Ensure positional encodings are correctly sized
        pe = self.pe[:, :seq_len, :].to(x.device)
        
        # Add positional encodings to input
        x = x + pe
        return x
    
class EncoderDecoderTransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(EncoderDecoderTransformerModel, self).__init__()
        
        # Embedding layers for source and target languages
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer network
        self.transformer = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, 
                                       batch_first=True)  # Setting batch_first=True for batch-first input
        
        # Final linear layer to project the output to target vocabulary size
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # src: (batch_size, src_seq_len)
        # tgt: (batch_size, tgt_seq_len)
        
        # Embedding and scaling by sqrt(d_model)
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        # src: (batch_size, src_seq_len, d_model)
        # tgt: (batch_size, tgt_seq_len, d_model)
        
        # Adding positional encoding
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        # src: (batch_size, src_seq_len, d_model)
        # tgt: (batch_size, tgt_seq_len, d_model)
        
        # Passing through the Transformer
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, 
                                  memory_mask=memory_mask, src_key_padding_mask=src_key_padding_mask, 
                                  tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        # output: (batch_size, tgt_seq_len, d_model)
        
        # Final linear layer
        output = self.fc_out(output)
        # output: (batch_size, tgt_seq_len, tgt_vocab_size)
        return output
    
    def generate_square_subsequent_mask(self, size):
        # Generate a mask for the target sequence to prevent attention to future tokens
        mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # mask: (size, size)
        return mask
    
    def create_padding_mask(self, seq):
        # Create a padding mask to ignore padded tokens (assuming padding token is 0)
        return (seq == 0)  # shape: (batch_size, seq_len)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.qkv_linear = nn.Linear(d_model, 3 * d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        qkv = self.qkv_linear(x)  # (batch_size, seq_len, 3 * d_model)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (batch_size, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        out = torch.matmul(attention, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        out = out.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        out = out.view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)
        
        # Final linear layer
        out = self.fc_out(out)  # (batch_size, seq_len, d_model)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = F.relu(self.fc1(x))  # (batch_size, seq_len, d_ff)
        x = self.dropout(x)  # (batch_size, seq_len, d_ff)
        x = self.fc2(x)  # (batch_size, seq_len, d_model)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: (batch_size, seq_len, d_model)
        attn_out = self.self_attn(x, mask)  # (batch_size, seq_len, d_model)
        x = x + self.dropout(attn_out)  # (batch_size, seq_len, d_model)
        x = self.norm1(x)  # (batch_size, seq_len, d_model)
        
        ff_out = self.ff(x)  # (batch_size, seq_len, d_model)
        x = x + self.dropout(ff_out)  # (batch_size, seq_len, d_model)
        x = self.norm2(x)  # (batch_size, seq_len, d_model)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1, max_len=5000, device='cpu'):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)  # (batch_size, seq_len, d_model)

        # Create a causal mask for self-attention
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)

        for layer in self.layers:
            x = layer(x, mask)  # (batch_size, seq_len, d_model)
        
        x = self.fc_out(x)  # (batch_size, seq_len, vocab_size)
        return x
    
    def infer(self, x, length=2048):
        with torch.no_grad():
            for step in range(length):
                output = self.forward(x)
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((x, predict), dim=-1)

                x = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids
    
class TransformerDecoder_NOPE(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1, max_len=5000, device='cpu'):
        super(TransformerDecoder_NOPE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        # x = self.pos_encoding(x)  # (batch_size, seq_len, d_model)

        # Create a causal mask for self-attention
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)

        for layer in self.layers:
            x = layer(x, mask)  # (batch_size, seq_len, d_model)
        
        x = self.fc_out(x)  # (batch_size, seq_len, vocab_size)
        return x
    
    def infer(self, x, length=2048):
        with torch.no_grad():
            for step in range(length):
                output = self.forward(x)
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((x, predict), dim=-1)

                x = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids

        
class GPT2Model(nn.Module):
    def __init__(self, vocab_size=140, n_embd=768, n_layer=12, n_head=12):
        super(GPT2Model, self).__init__()
        self.configuration = GPT2Config(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, bos_token_id=2, eos_token_id=1)
        self.model = GPT2LMHeadModel(self.configuration)
        
        # self.optimizer = Adam(self.model.parameters(), lr=3e-5)
        # self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)
    
    def forward(self, input_ids, labels=None):
        attention_mask = self.make_mask(input_ids)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output
    
    def make_mask(self, input_ids):
        attention_mask = (input_ids != 0).long()
        return attention_mask
    
    def infer(self, input_ids, length=2048):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        if len(input_ids.shape) > 2:
            raise Exception
        
        if length > 2048:
            print("Max Length is 2048. Change Length Auto to 2048")
            length = 2048
        
        with torch.no_grad():
            for step in range(length):
                output = self.forward(input_ids)
                output = torch.argmax(output.logits, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((input_ids, predict), dim=-1)

                input_ids = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids
    
    
class BARTModel(nn.Module):
    def __init__(self, vocab_size=140):
        super(BARTModel, self).__init__()
        self.configuration = GPT2Config(vocab_size=vocab_size, bos_token_id=2, eos_token_id=1)
        self.model = GPT2LMHeadModel(self.configuration)
        
        # self.optimizer = Adam(self.model.parameters(), lr=3e-5)
        # self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)
    
    def forward(self, input_ids, labels=None):
        attention_mask = self.make_mask(input_ids)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output
    
    def make_mask(self, input_ids):
        attention_mask = (input_ids != 0).long()
        return attention_mask
    
    def infer(self, input_ids, length=2048):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        if len(input_ids.shape) > 2:
            raise Exception
        
        if length > 2048:
            print("Max Length is 2048. Change Length Auto to 2048")
            length = 2048
        
        with torch.no_grad():
            for step in range(length):
                output = self.forward(input_ids)
                output = torch.argmax(output.logits, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((input_ids, predict), dim=-1)

                input_ids = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids
    

class RQTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, book_size=3, dropout=0.1, max_len=5000, device='cpu'):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.book_size = book_size
        self.book_pos_embed = nn.Embedding(book_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, device)
        
        self.Spatial_Layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        padding = remainder_to_mult(seq_len, self.book_size)
        x = F.pad(x, (0, padding), value = 0)
        x = rearrange(x, 'b (s d) -> b s d', d = self.book_size)
        
        x = self.embedding(x)  # (batch_size, seq_len_Q, book_size, d_model)
        book_pos = self.book_pos_embed(torch.arange(book_size, device=device))
        x = x + book_pos
        
        # average book token
        x = reduce(x, 'b s d f -> b s f', 'mean')
        x = self.pos_encoding(x)  # (batch_size, seq_len, d_model)

        # Create a causal mask for self-attention
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)

        for layer in self.Spatial_Layers:
            x = layer(x, mask)  # (batch_size, seq_len, d_model)
        
        
        x = self.fc_out(x)  # (batch_size, seq_len, vocab_size)
        return x
    
    def infer(self, x, length=2048):
        with torch.no_grad():
            for step in range(length):
                output = self.forward(x)
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((x, predict), dim=-1)

                x = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids
    
def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult