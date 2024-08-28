import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import T5Config, T5ForConditionalGeneration, AdamW
from transformers import BertConfig, BertModel, BertLMHeadModel, BartConfig, BartModel

class ChordLSTM(nn.Module):
    def __init__(self, vocab_size=140, embedding_dim=2048, hidden_dim=512, num_layers=5):
        super(ChordLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.epsilon = 1e-9
        self.optimizer = optim.AdamW(self.parameters(), lr=0.01, eps=1e-9)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=0.001)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        logits = self.fc(output)
        return logits
    
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
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((input_ids, predict), dim=-1)

                input_ids = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids
    
class ChordLSTM_Q(nn.Module):
    def __init__(self, vocab_size=140, embedding_dim=2048, hidden_dim=512, num_layers=5):
        super(ChordLSTM_Q, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.epsilon = 1e-9
        self.optimizer = optim.AdamW(self.parameters(), lr=0.007, eps=1e-9)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.96 ** epoch)
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=0.001)

    def forward(self, x):
        embedded = self.embedding(x)
        batch_size, seq_length, embed_dim = embedded.size()
        # print(embedded.shape)
        avg_stack = []
        for i in range(0,seq_length,3):
            # Average embeddings over the window
            # print(embedded[:,i:i+3,:].shape)
            avg_embed = embedded[:,i:i+3,:].mean(dim=1)
            avg_stack.append(avg_embed)
            # print(avg_embed.shape)
            # break
        
        avg_output = torch.stack(avg_stack, dim=1)
        # print(avg_output.shape)
        output, (hidden, cell) = self.lstm(avg_output)
        # print(output.shape)
        upsampled_out = output.transpose(1, 2)
        upsampled_out = F.interpolate(upsampled_out, size=seq_length, mode='linear', align_corners=True)
        upsampled_out = upsampled_out.transpose(1, 2)
        # print(upsampled_out.shape)
        logits = self.fc(upsampled_out)
        return logits
    
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
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((input_ids, predict), dim=-1)

                input_ids = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super(PositionalEncoding, self).__init__()

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return token_embedding + self.pos_encoding[:,:token_embedding.size(1)]


class ChordTransformer(nn.Module):
    # Constructor
    def __init__( self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, ):
        super(ChordTransformer, self).__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=5000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )
        self.out = nn.Linear(dim_model, num_tokens)
        
        # UTILS
        self.epsilon = 1e-9
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001, eps=1e-9)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=1e-9)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
    # def forward(self, src, src_mask):
        # Src, Tgt size 는 반드시 (batch_size, src sequence length) 여야 합니다.

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        output = self.out(transformer_out)
        
        # src = self.embedding(src) * math.sqrt(self.dim_model)
        # src = self.positional_encoder(src)
        
        # output = self.transformer(src, src, src_mask)
        # output = self.out(output)

        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=next(self.parameters()).device) * float('-inf'), diagonal=1)
        return mask

    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        return (matrix == pad_token)
    
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
                # src_mask = self.generate_square_subsequent_mask(input_ids.size(1))                
                inputs = input_ids[:,:]
                targets = input_ids[:, 1:]
                
                sequence_length = targets.size(1)
                tgt_mask = self.get_tgt_mask(sequence_length)
                
                output = self.forward(input_ids, targets, tgt_mask)
            
                # output = self.forward(input_ids, src_mask)
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((input_ids, predict), dim=-1)

                input_ids = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids

    
# Initialize the model
# model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)


class ChordBERT(nn.Module):
    def __init__(self, vocab_size=140, embedding_dim=2048, hidden_dim=512, num_layers=5):
        super(ChordBERT, self).__init__()
        self.configuration = BertConfig(vocab_size=vocab_size, max_position_embeddings=1024)
        self.model = BertModel(self.configuration)
        self.fc = nn.Linear(768, vocab_size)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.00001, eps=1e-9)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)
    
    def forward(self, input_ids):
        attention_mask = self.make_mask(input_ids)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        output = self.fc(output)
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
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((input_ids, predict), dim=-1)

                input_ids = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids
    
    
class ChordBART(nn.Module):
    def __init__(self, vocab_size=140, embedding_dim=2048, hidden_dim=512, num_layers=5):
        super(ChordBART, self).__init__()
        self.configuration = BartConfig(vocab_size=vocab_size, d_model=512)
        self.model = BartModel(self.configuration)
        # self.configuration = BertConfig(vocab_size=vocab_size)
        # self.model = BertModel(self.configuration)
        self.fc = nn.Linear(512, vocab_size)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.00001, eps=1e-9)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        attention_mask = self.make_mask(input_ids)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        output = self.fc(output)
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
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((input_ids, predict), dim=-1)

                input_ids = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids
    
    
    
import torch
import torch.nn.functional as F
from torch import nn, einsum
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops_exts import rearrange_with_anon_dims
from einops import rearrange, reduce, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# helper classes

def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        h, device = self.heads, x.device

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        mask_value = -torch.finfo(sim.dtype).max
        mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(mask, mask_value)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

# main class

class RQTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_spatial_seq_len,
        depth_seq_len,
        spatial_layers,
        depth_layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0
    ):
        super().__init__()
        self.dim = dim
        self.max_spatial_seq_len = max_spatial_seq_len
        self.depth_seq_len = depth_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.spatial_start_token = nn.Parameter(torch.randn(dim))

        self.spatial_pos_emb = nn.Embedding(max_spatial_seq_len + 1, dim) # account for a boundary case
        self.depth_pos_emb = nn.Embedding(depth_seq_len, dim)

        self.spatial_transformer = Transformer(
            dim = dim,
            layers = spatial_layers,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        self.depth_transformer = Transformer(
            dim = dim,
            layers = depth_layers,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        self.to_logits = nn.Linear(dim, num_tokens)
        self.pad_id = pad_id
        self.optimizer = optim.AdamW(self.parameters(), lr=0.0003, eps=1e-9)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    def generate(self, length=100, prime = None, filter_thres = 0.9, temperature = 1., default_batch_size = 1):
        total_seq_len = self.depth_seq_len * self.max_spatial_seq_len
        device = next(self.parameters()).device

        if not exists(prime):
            prime = torch.empty((default_batch_size, 0), dtype = torch.long, device = device)

        seq = prime

        for _ in range(length):
            logits = self.forward(seq)[:, -1]
            logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
            seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)

        # return rearrange(seq, 'b (s d) -> b s d', d = self.depth_seq_len)
        return seq

    def forward_empty(self, batch_size):
        # take care of special case
        # where you sample from input of 0 (start token only)

        # self.spatial_start_token = tensor([-0.2637,  0.5506, -0.8933, -0.4647], requires_grad=True)
        # Input 모양 바꿔주는것
        spatial_tokens = repeat(self.spatial_start_token, 'd -> b 1 d', b = batch_size)
        
        # 그냥 바로 Token IDX 무시하고 랜덤 Embedding Transformer에 넣어주는것
        depth_tokens = self.spatial_transformer(spatial_tokens)
        depth_tokens = self.depth_transformer(depth_tokens)
        
        # 바로 Transformer 돌려서 token 만큼 hidden에 logit 뽑기
        return self.to_logits(depth_tokens)

    def forward(self, ids, return_loss = False):
        assert ids.ndim in {2, 3}
        # 원래 3차원 (RQ니까 bsz, seq, quant codebook) 인데 flat 시켜서 (bsz, seq*Q) 확인용
        flattened_dim = ids.ndim == 2
        ids_orig_ndim = ids.ndim

        # 아무것도 인풋 없이 첫번째 토큰으로만 forward하는 경우
        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        if flattened_dim:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            
            #padding -> codebook 남은것 ex) 4이고 input이 13이면 마지막에 1개 남아서 3이 리턴
            padding = remainder_to_mult(seq_len, self.depth_seq_len)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = rearrange(ids, 'b (s d) -> b s d', d = self.depth_seq_len)
        else:
            # seq len을 seq_len * codebook 으로 바꿔서 flatten한 숫자
            seq_len = ids.shape[1] * ids.shape[2]

        b, space, depth, device = *ids.shape, ids.device
        # 1024
        assert space <= (self.max_spatial_seq_len + 1), 'spatial dimension is greater than the max_spatial_seq_len set'
        # 4
        assert depth == self.depth_seq_len, 'depth dimension must be equal to depth_seq_len'

        # get token embeddings

        tokens = self.token_emb(ids)

        spatial_pos = self.spatial_pos_emb(torch.arange(space, device = device))
        depth_pos = self.depth_pos_emb(torch.arange(depth, device = device))

        tokens_with_depth_pos = tokens + depth_pos

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions

        spatial_tokens = reduce(tokens_with_depth_pos, 'b s d f -> b s f', 'sum') + spatial_pos

        spatial_tokens = torch.cat((
            repeat(self.spatial_start_token, 'f -> b 1 f', b = b),
            spatial_tokens
        ), dim = -2)        

        spatial_tokens = self.spatial_transformer(spatial_tokens)

        spatial_tokens = rearrange(spatial_tokens, 'b s f -> b s 1 f')

        # spatial tokens become the start tokens of the depth dimension

        tokens_with_depth_pos = F.pad(tokens_with_depth_pos, (0, 0, 0, 0, 0, 1), value = 0.)

        depth_tokens = torch.cat((spatial_tokens, tokens_with_depth_pos), dim = -2)

        depth_tokens = rearrange(depth_tokens, '... n d -> (...) n d')

        depth_tokens = self.depth_transformer(depth_tokens)

        depth_tokens = rearrange(depth_tokens, '(b s) d f -> b s d f', b = b)

        
        logits = self.to_logits(depth_tokens)

        logits = rearrange(logits, 'b ... f -> b (...) f')

        logits = logits[:, :(seq_len + 1)]

        if not return_loss:

            logits = logits[:, 1:]

            if flattened_dim:

                a = rearrange(logits, 'b ... n -> b (...) n')

                return rearrange(logits, 'b ... n -> b (...) n')

            return logits

        logits = logits[:, :-1]

        preds = rearrange(logits, 'b ... c -> b c (...)')
        labels = rearrange(ids, 'b s d -> b (s d)')

        labels = labels[:,:seq_len]

        
        loss = F.cross_entropy(preds, labels, ignore_index = self.pad_id)
        return loss, preds

# model = RQTransformer(
#     num_tokens = 50,             # number of tokens, in the paper they had a codebook size of 16k
#     dim = 512,                      # transformer model dimension
#     max_spatial_seq_len = 1024,     # maximum positions along space
#     depth_seq_len = 3,              # number of positions along depth (residual quantizations in paper)
#     spatial_layers = 8,             # number of layers for space
#     depth_layers = 4,               # number of layers for depth
#     dim_head = 64,                  # dimension per head
#     heads = 8,                      # number of attention heads
# )

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops_exts import rearrange_with_anon_dims
from einops import rearrange, reduce, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# helper classes

def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        h, device = self.heads, x.device

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        mask_value = -torch.finfo(sim.dtype).max
        mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(mask, mask_value)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class RQ_DelayTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_spatial_seq_len,
        depth_seq_len,
        spatial_layers,
        depth_layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0
    ):
        super().__init__()
        self.dim = dim
        self.max_spatial_seq_len = max_spatial_seq_len
        self.depth_seq_len = depth_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.spatial_start_token = nn.Parameter(torch.randn(dim))

        self.spatial_pos_emb = nn.Embedding(max_spatial_seq_len + 1, dim) # account for a boundary case
        self.depth_pos_emb = nn.Embedding(depth_seq_len, dim)

        self.spatial_transformer = Transformer(
            dim = dim,
            layers = spatial_layers,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        self.depth_transformer = Transformer(
            dim = dim,
            layers = depth_layers,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        self.to_logits = nn.Linear(dim, num_tokens)
        self.pad_id = pad_id
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    def generate(self, length=100, prime = None, filter_thres = 0.9, temperature = 1., default_batch_size = 1):
        total_seq_len = self.depth_seq_len * self.max_spatial_seq_len
        device = next(self.parameters()).device

        if not exists(prime):
            prime = torch.empty((default_batch_size, 0), dtype = torch.long, device = device)

        seq = prime

        for _ in range(length):
            logits = self.forward(seq)[:, -1]
            logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
            seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)
        # print(seq)
        # print(seq.shape)
        return rearrange(seq, 'b (s d) -> b s d', d = self.depth_seq_len)
        return seq

    def forward_empty(self, batch_size):
        # take care of special case
        # where you sample from input of 0 (start token only)

        # self.spatial_start_token = tensor([-0.2637,  0.5506, -0.8933, -0.4647], requires_grad=True)
        # Input 모양 바꿔주는것
        spatial_tokens = repeat(self.spatial_start_token, 'd -> b 1 d', b = batch_size)
        
        # 그냥 바로 Token IDX 무시하고 랜덤 Embedding Transformer에 넣어주는것
        depth_tokens = self.spatial_transformer(spatial_tokens)
        depth_tokens = self.depth_transformer(depth_tokens)
        
        # 바로 Transformer 돌려서 token 만큼 hidden에 logit 뽑기
        return self.to_logits(depth_tokens)

    def forward(self, ids, return_loss = False):
        assert ids.ndim in {2, 3}
        # 원래 3차원 (RQ니까 bsz, seq, quant codebook) 인데 flat 시켜서 (bsz, seq*Q) 확인용
        flattened_dim = ids.ndim == 2
        ids_orig_ndim = ids.ndim

        # 아무것도 인풋 없이 첫번째 토큰으로만 forward하는 경우
        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        if flattened_dim:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            
            #padding -> codebook 남은것 ex) 4이고 input이 13이면 마지막에 1개 남아서 3이 리턴
            padding = remainder_to_mult(seq_len, self.depth_seq_len)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = rearrange(ids, 'b (s d) -> b s d', d = self.depth_seq_len)
        else:
            # seq len을 seq_len * codebook 으로 바꿔서 flatten한 숫자
            seq_len = ids.shape[1] * ids.shape[2]

        b, space, depth, device = *ids.shape, ids.device
        # 1024
        assert space <= (self.max_spatial_seq_len + 1), 'spatial dimension is greater than the max_spatial_seq_len set'
        # 4
        assert depth == self.depth_seq_len, 'depth dimension must be equal to depth_seq_len'

        # get token embeddings

        tokens = self.token_emb(ids)

        spatial_pos = self.spatial_pos_emb(torch.arange(space, device = device))
        depth_pos = self.depth_pos_emb(torch.arange(depth, device = device))

        tokens_with_depth_pos = tokens + depth_pos

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions

        spatial_tokens = reduce(tokens_with_depth_pos, 'b s d f -> b s f', 'sum') + spatial_pos

        spatial_tokens = torch.cat((
            repeat(self.spatial_start_token, 'f -> b 1 f', b = b),
            spatial_tokens
        ), dim = -2)        

        spatial_tokens = self.spatial_transformer(spatial_tokens)

        spatial_tokens = rearrange(spatial_tokens, 'b s f -> b s 1 f')

        # spatial tokens become the start tokens of the depth dimension

        tokens_with_depth_pos = F.pad(tokens_with_depth_pos, (0, 0, 0, 0, 0, 1), value = 0.)

        depth_tokens = torch.cat((spatial_tokens, tokens_with_depth_pos), dim = -2)

        depth_tokens = rearrange(depth_tokens, '... n d -> (...) n d')

        depth_tokens = self.depth_transformer(depth_tokens)

        depth_tokens = rearrange(depth_tokens, '(b s) d f -> b s d f', b = b)

        logits = self.to_logits(depth_tokens)
        logits = rearrange(logits, 'b ... f -> b (...) f')
        logits = logits[:, :(seq_len + 1)]

        if not return_loss:
            logits = logits[:, 1:]

            if flattened_dim:
                return rearrange(logits, 'b ... n -> b (...) n')

            return logits

        logits = logits[:, :-1]
        
        preds = rearrange(logits, 'b ... c -> b c (...)')
        labels = rearrange(ids, 'b s d -> b (s d)')

        loss = F.cross_entropy(preds, labels, ignore_index = self.pad_id)
        return loss, preds


model = RQTransformer(
    num_tokens = 50,             # number of tokens, in the paper they had a codebook size of 16k
    dim = 512,                      # transformer model dimension
    max_spatial_seq_len = 1024,     # maximum positions along space
    depth_seq_len = 3,              # number of positions along depth (residual quantizations in paper)
    spatial_layers = 8,             # number of layers for space
    depth_layers = 4,               # number of layers for depth
    dim_head = 64,                  # dimension per head
    heads = 8,                      # number of attention heads
)




import torch
import torch.nn as nn
import torch.optim as optim
import math

class BPE_TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, num_layers, output_dim, dropout=0.1):
        super(BPE_TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.pos_encoder = BPE_PositionalEncoding(embed_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(embed_dim, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=0.003, eps=1e-9)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        # Transpose src to shape [seq_len, bsz, embed_dim] for transformer
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, src_mask)
        # Transpose back to shape [bsz, seq_len, embed_dim]
        output = output.transpose(0, 1)
        output = self.fc_out(output)
        return output
    
    def infer(self, input_ids, length=2048, device='cpu'):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        if len(input_ids.shape) > 2:
            raise Exception
        
        if length > 2048:
            print("Max Length is 2048. Change Length Auto to 2048")
            length = 2048
            
        with torch.no_grad():
            for step in range(length):
                src_mask = self.generate_square_subsequent_mask(input_ids.size(1)).to(device)
                output = self.forward(input_ids, src_mask)
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((input_ids, predict), dim=-1)

                input_ids = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        

class BPE_PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(BPE_PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
