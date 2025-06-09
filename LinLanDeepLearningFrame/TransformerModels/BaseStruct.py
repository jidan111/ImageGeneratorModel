from . import *


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head_num, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % head_num == 0, f"输入维度{d_model}必须能被平均分配到每一个头{head_num}上，{d_model}%{head_num}!=0"
        self.d_model = d_model
        self.head_num = head_num
        self.d_k = d_model // head_num
        self.sqrt_dk = math.sqrt(self.d_k)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        :param Q: [batch_size, head_nums, seq_len, d_k]
        :param K:[batch_size, head_nums, seq_len, d_k]
        :param V:[batch_size, head_nums, seq_len, d_k]
        :param mask:[batch_size, 1, 1, seq_len]
        :return:
        """
        # attn_shape:[batch_size, head_num, seq_len, seq_len]
        attn_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.sqrt_dk
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, 1e-9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, V)
        return out

    def forward(self, Q, K, V, mask=None):
        """
        :param Q: [batch_size, seq_len, d_model]
        :param K: [batch_size, seq_len, d_model]
        :param V: [batch_size, seq_len, d_model]
        :param mask: [batch_size, seq_len, d_model]
        :return:
        """
        batch_size, seq_len, d_model = Q.shape
        q = self.Wq(Q).view(batch_size, seq_len, self.head_num, self.d_k).permute(0, 2, 1, 3)
        v = self.Wv(V).view(batch_size, seq_len, self.head_num, self.d_k).permute(0, 2, 1, 3)
        k = self.Wk(K).view(batch_size, seq_len, self.head_num, self.d_k).permute(0, 2, 1, 3)
        attn_output = self.scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, d_model)
        return attn_output  # [batch_size, seq_len, d_model]


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, head_num, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model=d_model, head_num=head_num, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, head_num, d_ff=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, head_num=head_num, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, head_num=head_num, dropout=dropout)
        self.ffc = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        self_attn = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn))
        cross_attn = self.cross_attn(Q=x, K=encoder_out, V=encoder_out, mask=src_mask)
        x = self.norm2(x + self.dropout2(cross_attn))
        ffc = self.ffc(x)
        x = self.norm3(x + self.dropout3(ffc))
        return x
