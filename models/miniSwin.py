import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PixelWiseDotProduct


# -----------------------------
# Window Partition / Reverse
# -----------------------------
def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C,
               H // window_size, window_size,
               W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = windows.view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B,
                     H // window_size,
                     W // window_size,
                     window_size,
                     window_size,
                     -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, -1, H, W)
    return x


# -----------------------------
# Swin Block
# -----------------------------
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=7):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # pad if needed
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h))

        Hp, Wp = x.shape[2], x.shape[3]

        windows = window_partition(x, self.window_size)
        windows = self.norm1(windows)

        attn_windows, _ = self.attn(windows, windows, windows)
        windows = windows + attn_windows

        windows = windows + self.mlp(self.norm2(windows))

        x = window_reverse(windows, self.window_size, Hp, Wp)

        return x[:, :, :H, :W]


# -----------------------------
# Swin Replacement for mViT
# -----------------------------
class SwinBins(nn.Module):
    def __init__(self,
                 in_channels,
                 n_query_channels=128,
                 dim_out=256,
                 embedding_dim=128,
                 norm='linear'):
        super().__init__()

        self.norm = norm
        self.n_query_channels = n_query_channels

        self.conv_embed = nn.Conv2d(in_channels, embedding_dim, 3, 1, 1)

        self.swin = nn.Sequential(
            SwinBlock(embedding_dim),
            SwinBlock(embedding_dim)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, dim_out)
        )

        self.query_conv = nn.Conv2d(embedding_dim,
                                    n_query_channels,
                                    kernel_size=1)

        self.dot_product_layer = PixelWiseDotProduct()

    def forward(self, x):

        B, C, H, W = x.shape

        x_embed = self.conv_embed(x)
        swin_features = self.swin(x_embed)

        # ---- bin width regression ----
        pooled = self.global_pool(swin_features).view(B, -1)
        y = self.regressor(pooled)

        if self.norm == 'linear':
            y = torch.relu(y) + 0.1
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), None
        else:
            y = torch.sigmoid(y)

        y = y / y.sum(dim=1, keepdim=True)

        # ---- range attention maps ----
        queries = self.query_conv(swin_features)  # B,128,H,W
        queries = queries.flatten(2).permute(0, 2, 1)  # B,HW,128

        range_attention_maps = self.dot_product_layer(
            swin_features,
            queries[:, :self.n_query_channels, :]
        )

        return y, range_attention_maps
