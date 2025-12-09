import torch
from torch import nn
import torch.nn.functional as F
import math
from torch import Tensor


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size,d_model):
        super(TokenEmbedding,self).__init__(vocab_size,d_model,padding_idx=1)
class PositionalEncoding(nn.Module):
   def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 创建 (max_len, d_model) 的全0矩阵
        pe = torch.zeros(max_len, d_model)

        # pos = [0,1,2,...,max_len-1]^T  shape = (max_len,1)
        position = torch.arange(0, max_len).unsqueeze(1)

        # div_term = 10000^{2i/d_model}
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        # shape = (1, max_len, d_model)

   def forward(self, x):
        """
        x: (batch_size, seq_len)
        return: (1, seq_len, d_model)
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device): # 必须有 device
        super(TransformerEmbedding, self).__init__()
        self.device = device

        # 构造时就将子层移到 device
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.drop_out = nn.Dropout(drop_prob)

    def forward(self, x):
        # 输入 x 必须移到 device
        x = x.to(self.device)
        tok = self.tok_emb(x)
        pos = self.pos_emb(x)
        return self.drop_out(tok + pos)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch, len_q, _ = q.shape
        batch, len_k, _ = k.shape
        batch, len_v, _ = v.shape

        # 线性映射
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # 分头
        q = q.view(batch, len_q, self.n_head, self.d_head).permute(0, 2, 1, 3)
        k = k.view(batch, len_k, self.n_head, self.d_head).permute(0, 2, 1, 3)
        v = v.view(batch, len_v, self.n_head, self.d_head).permute(0, 2, 1, 3)

        # Attention score
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Mask（自动广播）
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        # softmax + v
        attn = self.softmax(score)
        context = attn @ v  # (batch, n_head, len_q, d_head)

        # 多头 concat
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch, len_q, self.d_model)

        # 输出线性
        out = self.w_combine(context)
        return out
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
#
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # 计算最后一个维度的 mean 和 var
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        # 标准化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 缩放 + 平移
        return self.gamma * x_norm + self.beta
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head Attention
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # Feed Forward Network
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x


class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, max_len, d_model,
                 ffn_hidden, n_head, n_layer, dropout=0.1, device=None):  # 新增 device 参数
        super(Encoder, self).__init__()
        self.device = device

        self.embedding = TransformerEmbedding(
            enc_vocab_size,
            d_model,
            max_len,
            dropout,
            device  # 将 device 传入
        )

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, ffn_hidden, n_head, dropout)
            for _ in range(n_layer)
        ])
        # 模型已经通过 .to(device) 移到 GPU，因此不需要在这里重复

    def forward(self, x, s_mask):
        # x 在 embedding 内部已移至 device
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x


class DecoderLayer(nn.Module):
        def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
            super().__init__()
            self.attention1 = MultiHeadAttention(d_model, n_head)
            self.norm1 = LayerNorm(d_model)
            self.dropout1 = nn.Dropout(drop_prob)

            self.cross_attention = MultiHeadAttention(d_model, n_head)
            self.norm2 = LayerNorm(d_model)
            self.dropout2 = nn.Dropout(drop_prob)

            self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
            self.norm3 = LayerNorm(d_model)
            self.dropout3 = nn.Dropout(drop_prob)

        def forward(self, dec, enc, t_mask, s_mask):
            # ---- 自注意力 ----
            _x = dec
            x = self.attention1(dec, dec, dec, t_mask)
            x = self.dropout1(x)
            x = self.norm1(x + _x)

            # ---- 交叉注意力 ----
            _x = x
            x = self.cross_attention(x, enc, enc, s_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)

            # ---- FFN ----
            _x = x
            x = self.ffn(x)
            x = self.dropout3(x)
            x = self.norm3(x + _x)

            return x


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden,
                 n_head, n_layer, drop_prob, device=None):  # 新增 device 参数
        super().__init__()

        self.embedding = TransformerEmbedding(
            dec_voc_size, d_model, max_len, drop_prob, device
        )

        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layer)
            ]
        )

        self.fc = nn.Linear(d_model, dec_voc_size)

    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)
        dec = self.fc(dec)
        return dec


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size,
                 d_model, max_len, n_heads, ffn_hidden, n_layers, drop_prob, device):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden,
                               n_heads, n_layers, drop_prob, device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden,
                               n_heads, n_layers, drop_prob, device)

    def make_src_mask(self, src):
        # 源序列的padding mask: (B, 1, 1, L_src)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # 目标序列的mask: padding mask + causal mask
        batch_size, trg_len = trg.shape

        # Padding mask: (B, 1, 1, L_trg)
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # Causal mask: (1, 1, L_trg, L_trg)
        trg_causal_mask = torch.tril(torch.ones(trg_len, trg_len)).bool().unsqueeze(0).unsqueeze(1)

        # 合并mask: (B, 1, L_trg, L_trg)
        trg_mask = trg_pad_mask & trg_causal_mask.to(self.device)
        return trg_mask

    def forward(self, src, trg):
        # 确保输入在正确设备上
        src, trg = src.to(self.device), trg.to(self.device)

        # 生成mask
        src_mask = self.make_src_mask(src)  # (B, 1, 1, L_src)
        trg_mask = self.make_trg_mask(trg)  # (B, 1, L_trg, L_trg)

        # 编码器-解码器
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_out, trg_mask, src_mask)
        return out
