import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute scaled dot-product attention:
      Attn(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V.
    Args:
        q: (batch, heads, len_q, d_k)
        k: (batch, heads, len_k, d_k)
        v: (batch, heads, len_k, d_v) (here d_v = d_k)
        mask: optional (batch, 1, len_q, len_k) or (batch, len_q, len_k) where False/0 indicates masked positions.
    Returns:
        (batch, heads, len_q, d_v)
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def subsequent_mask(size: int) -> torch.Tensor:
    """
    Returns shape: (1, size, size)
    True where attention is allowed.
    """
    return torch.tril(torch.ones(size, size, dtype=torch.bool)).unsqueeze(0)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        query: (batch, len_q, d_model)
        key:   (batch, len_k, d_model)
        value: (batch, len_k, d_model)
        mask:  (batch, 1, len_q, len_k) with 0's in positions to mask
        returns: (batch, len_q, d_model)
        """
        batch_size = query.size(0)
        # Linear projections
        Q = self.w_q(query)  # (batch, len_q, d_model)
        K = self.w_k(key)    # (batch, len_k, d_model)
        V = self.w_v(value)  # (batch, len_k, d_model)
        # Split into heads 
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)  # (batch, heads, len_q, d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)  # (batch, heads, len_k, d_k)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)  # (batch, heads, len_k, d_k)
        # Scaled dot-product attention on all heads
        attn_out = scaled_dot_product_attention(Q, K, V, mask)  # (batch, heads, len_q, d_k)
        # Combine heads
        attn_out = attn_out.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)  # (batch, len_q, d_model)
        return self.dropout(self.w_o(attn_out))  # (batch, len_q, d_model)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims: cos
        self.register_buffer('pe', pe)  # not a parameter but saved in state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)  # broadcast to batch
        return self.dropout(x)


class EncoderLayer(nn.Module):

    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch, src_len, d_model)
        src_mask: (batch, 1, 1, src_len)
        returns: (batch, src_len, d_model)
        """
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (batch, tgt_len, d_model)
        memory: (batch, src_len, d_model)
        src_mask: (batch, 1, 1, src_len)
        tgt_mask: (batch, 1, tgt_len, tgt_len)
        returns: (batch, tgt_len, d_model)
        """
        self_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = x + self.dropout(self_out)
        x = self.norm1(x)

        enc_out = self.enc_attn(x, memory, memory, mask=src_mask)
        x = x + self.dropout(enc_out)
        x = self.norm2(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm3(x)
        return x


class Encoder(nn.Module):

    def __init__(self, num_layers: int = 6, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):

    def __init__(self, num_layers: int = 6, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor, 
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)

        self.output_linear = nn.Linear(d_model, tgt_vocab)
        self.scale = math.sqrt(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        src_emb = self.src_embedding(src) * self.scale  # (batch, src_len, d_model)
        src_emb = self.pos_encoder(src_emb)  # (batch, src_len, d_model)
        return self.encoder(src_emb, src_mask)  # (batch, src_len, d_model)
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        tgt_emb = self.tgt_embedding(tgt) * self.scale  # (batch, tgt_len, d_model)
        tgt_emb = self.pos_encoder(tgt_emb)  # (batch, tgt_len, d_model)
        return self.decoder(tgt_emb, memory, src_mask, tgt_mask)  # (batch, tgt_len, d_model)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        src: source sequence (batch, src_len)
        tgt: target sequence (batch, tgt_len)
        src_mask: source mask to ignore padding tokens (batch, 1, 1, src_len)
        tgt_mask: target mask to prevent seeing future tokens (batch, 1, tgt_len, tgt_len)
        returns: log-probabilities (batch, tgt_len, tgt_vocab)
        """
        memory = self.encode(src, src_mask)  # (batch, src_len, d_model)
        out = self.decode(tgt, memory, src_mask, tgt_mask)  # (batch, tgt_len, d_model)
        logits = self.output_linear(out)  # (batch, tgt_len, tgt_vocab)
        return F.log_softmax(logits, dim=-1)  # (batch, tgt_len, tgt_vocab)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_len: int,
        src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        src: (batch, src_len)
        start_token_id: <SOS> token id
        end_token_id: <EOS> token id
        max_len: maximum generated length INCLUDING the start token
        src_mask: (batch, 1, 1, src_len) or None
        returns: generated token ids: (batch, generated_len)
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)

        memory = self.encode(src, src_mask)
        generated = torch.full(
            (batch_size, 1),
            start_token_id,
            dtype=torch.long,
            device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_mask = subsequent_mask(generated.size(1)).to(device)  # (1, len, len)
            tgt_mask = tgt_mask.unsqueeze(1)  # (1, 1, len, len)
            out = self.decode(generated, memory, src_mask, tgt_mask)  # (batch, len, d_model)
            logits = self.output_linear(out)  # (batch, len, tgt_vocab)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (batch, 1)
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == end_token_id)
            if finished.all():
                break

        return generated
