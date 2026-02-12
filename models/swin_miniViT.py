import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PixelWiseDotProduct


# -----------------------------
# Window utilities
# -----------------------------
def window_partition(x, window_size):
    """
    x: (B, H, W, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    windows: (num_windows*B, window_size*window_size, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


# -----------------------------
# Swin Block (W-MSA only)
# -----------------------------
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.0):
        super().__init__()
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, H, W):
        """
        x: (B, HW, C)
        """
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # padding
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]

        windows = window_partition(x, self.window_size)
        attn_windows, _ = self.attn(windows, windows, windows)

        x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------
# Cross Attention (Queries ← Tokens)
# -----------------------------
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, queries, tokens):
        """
        queries: (B, Nq, C)
        tokens:  (B, HW, C)
        """
        q = self.norm(queries)
        k = self.norm(tokens)
        v = tokens
        out, _ = self.attn(q, k, v)
        return queries + out


# -----------------------------
# Swin MiniViT (AdaBins-compatible)
# -----------------------------
class SwinMiniViT(nn.Module):
    def __init__(
        self,
        in_channels,
        n_bins=128,
        embed_dim=128,
        dim_out=256,
        depth=4,
        num_heads=4,
        window_size=7,
        norm='linear'
    ):
        super().__init__()
        self.norm_type = norm

        # input projection
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # Swin blocks (no downsampling)
        self.swin_blocks = nn.ModuleList([
            SwinBlock(embed_dim, num_heads, window_size)
            for _ in range(depth)
        ])

        # learned bin queries
        self.query_embed = nn.Parameter(
            torch.randn(1, n_bins, embed_dim)
        )

        # cross attention
        self.cross_attn = CrossAttention(embed_dim, num_heads)

        # bin width regressor
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, dim_out)
        )

        self.dot_product_layer = PixelWiseDotProduct()

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, _, H, W = x.shape

        feat = self.input_proj(x)                 # (B, E, H, W)
        tokens = feat.flatten(2).transpose(1, 2) # (B, HW, E)

        for blk in self.swin_blocks:
            tokens = blk(tokens, H, W)

        queries = self.query_embed.repeat(B, 1, 1)
        queries = self.cross_attn(queries, tokens)

        bin_features = queries.mean(dim=1)
        y = self.regressor(bin_features)

        if self.norm_type == 'linear':
            y = F.relu(y) + 0.1
            y = y / y.sum(dim=1, keepdim=True)
        elif self.norm_type == 'softmax':
            y = torch.softmax(y, dim=1)
        else:
            y = torch.sigmoid(y)
            y = y / y.sum(dim=1, keepdim=True)

        range_attention_maps = self.dot_product_layer(feat, queries)
        return y, range_attention_maps
