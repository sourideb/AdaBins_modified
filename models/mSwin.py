import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Attention
from .layers import PixelWiseDotProduct


class SwinLikeBlock(nn.Module):
    """
    Resolution-agnostic Swin-style block (window attention)
    """
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, H, W):
        """
        x: (N, HW, C)
        """
        N, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)

        # ---- reshape to windows ----
        x = x.view(N, H, W, C)

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = x.shape[1], x.shape[2]

        x = x.view(
            N,
            Hp // self.window_size,
            self.window_size,
            Wp // self.window_size,
            self.window_size,
            C
        ).permute(0, 1, 3, 2, 4, 5).contiguous()

        windows = x.view(-1, self.window_size * self.window_size, C)

        # ---- window attention ----
        attn_windows = self.attn(windows)

        # ---- merge windows ----
        x = attn_windows.view(
            N,
            Hp // self.window_size,
            Wp // self.window_size,
            self.window_size,
            self.window_size,
            C
        ).permute(0, 1, 3, 2, 4, 5).contiguous()

        x = x.view(N, Hp, Wp, C)[:, :H, :W, :]
        x = x.reshape(N, H * W, C)

        # ---- FFN ----
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class mSwin(nn.Module):
    def __init__(
        self,
        in_channels,
        n_query_channels=128,
        dim_out=256,
        embedding_dim=128,
        norm='linear',
        depth=4,
        num_heads=4,
        window_size=7
    ):
        super().__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels

        self.input_proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=1)

        self.blocks = nn.ModuleList([
            SwinLikeBlock(
                dim=embedding_dim,
                num_heads=num_heads,
                window_size=window_size
            )
            for _ in range(depth)
        ])

        self.dot_product_layer = PixelWiseDotProduct()

        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, dim_out)
        )

    def forward(self, x):
        """
        x: (N, C, H, W)
        """
        N, _, H, W = x.shape

        x_proj = self.input_proj(x)  # (N, E, H, W)
        tokens = x_proj.flatten(2).transpose(1, 2)  # (N, HW, E)

        for blk in self.blocks:
            tokens = blk(tokens, H, W)

        regression_head = tokens.mean(dim=1)
        queries = tokens[:, :self.n_query_channels, :]

        range_attention_maps = self.dot_product_layer(x_proj, queries)

        y = self.regressor(regression_head)

        if self.norm == 'linear':
            y = F.relu(y) + 0.1
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)

        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps
