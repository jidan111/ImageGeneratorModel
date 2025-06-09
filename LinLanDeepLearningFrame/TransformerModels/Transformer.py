from .BaseStruct import *


class Transformer(nn.Module):
    def __init__(self, src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 head_num=8,
                 d_ff=2048,
                 dropout=0.1,
                 encoder_layer_num=6,
                 decoder_layer_num=6,
                 pos_max_length=5000):
        super(Transformer, self).__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=pos_max_length)
        self.encoder_layer = nn.ModuleList(
            [EncoderLayer(d_model=d_model, head_num=head_num, d_ff=d_ff, dropout=dropout)
             for _ in range(encoder_layer_num)]
        )
        self.decoder_layer = nn.ModuleList(
            [DecoderLayer(d_model=d_model, head_num=head_num, d_ff=d_ff, dropout=dropout)
             for _ in range(decoder_layer_num)]
        )
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for i in self.encoder_layer:
            x = i(x, src_mask)
        return x

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        x = self.tgt_embed(tgt)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for i in self.decoder_layer:
            x = i(x, encoder_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_out = self.encode(src=src, src_mask=src_mask)
        decoder_out = self.decode(tgt=tgt, encoder_output=encoder_out,
                                  src_mask=src_mask, tgt_mask=tgt_mask)
        out = self.final_linear(decoder_out)
        return out

    def generate_src_mask(self, src, pad_idx=0):
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, src_len]

    def generate_tgt_mask(self, tgt, pad_idx=0):
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]
        seq_len = tgt.size(1)
        subsequent_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(tgt.device)
        return tgt_pad_mask & subsequent_mask  # 结合padding mask和未来位置mask


if __name__ == "__main__":
    # 示例参数
    src_vocab_size = 1000
    tgt_vocab_size = 2000
    batch_size = 32
    seq_len = 10

    # 初始化模型
    model = Transformer(src_vocab_size, tgt_vocab_size)

    # 模拟输入数据
    src = torch.randint(0, src_vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))

    # 生成mask
    src_mask = model.generate_src_mask(src)
    tgt_mask = model.generate_tgt_mask(tgt)

    # 前向传播
    output = model(src, tgt, src_mask, tgt_mask)
    print("输出尺寸:", output.size())  # 应为 [batch_size, seq_len, tgt_vocab_size]
