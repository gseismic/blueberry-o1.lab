
from .attention import Attention
from .mlp import FeedForward, MoEFeedForward
from .LMConfig import LMConfig
from ...nnblock.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LMConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        if args.use_moe:
            self.feed_forward = MOEFeedForward(args)
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )

    def forward(self, x, pos_cis, kv_cache=False):
        h = x + self.attention(self.attention_norm(x), pos_cis, kv_cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
