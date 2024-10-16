from ....config import nn, F
from .positional_encoding import PositionalEncoding
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, vocab_size,
                 seq_len=512, dropout=0.1, pe_base=10000):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        print(f'{embed_dim=}')
        print(f'{seq_len=}, type(seq_len)')
        self.embedding = nn.Embedding(vocab_size, embed_dim) 
        self.positional_encoding = PositionalEncoding(embed_dim, seq_len, pe_base=pe_base)
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, dropout)
        self.decoder = TransformerDecoder(num_layers, embed_dim, num_heads, ff_dim, dropout)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: [batch_size, seq_len]
        # tgt: [batch_size, seq_len]
        #print(f'{src.shape=}')
        #print(f'{tgt.shape=}')
        #print(f'{self.embedding(src).shape=}')

        # src, tgt:
        #   self.embedding: [batch_size, seq_len, embedding]
        #   self.positional_encoding: [1, seq_len, embed_dim]
        src_emb = self.embedding(src) + self.positional_encoding(src)
        tgt_emb = self.embedding(tgt) + self.positional_encoding(tgt)
        # src_emb, tgt_emb: [batch_size, seq_len, embedding]
        enc_output = self.encoder(src_emb, src_mask)
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)
        output = self.fc_out(dec_output)
        return output

