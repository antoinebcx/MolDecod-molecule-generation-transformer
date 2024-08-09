import torch
import torch.nn as nn
import math


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(RotaryPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.register_buffer('sin_pos', torch.sin(position * div_term))
        self.register_buffer('cos_pos', torch.cos(position * div_term))

    def forward(self, x):
        seq_len = x.size(1)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x = torch.cat([
            x1 * self.cos_pos[:seq_len] - x2 * self.sin_pos[:seq_len],
            x1 * self.sin_pos[:seq_len] + x2 * self.cos_pos[:seq_len]
        ], dim=-1)
        return self.dropout(x)


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout=0.1):
        super(DecoderOnlyTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = RotaryPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask)
        output = self.fc_out(output)
        return self.dropout(output)


def create_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask
