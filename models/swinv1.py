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

        windows = window_partition(x, self.window_size)  # (num_windows*B, ws*ws, C)

        # FIX 1: Correct pre-LN residual pattern.
        # Save the unnormalized shortcut BEFORE norm, apply norm only inside
        # the branch, then add back to the original unnormalized shortcut.
        # Previously: shortcut was already normalized before the residual add,
        # which breaks the gradient highway that skip connections provide.
        shortcut = windows
        attn_out, _ = self.attn(self.norm1(windows),
                                self.norm1(windows),
                                self.norm1(windows))
        windows = shortcut + attn_out                    # clean residual ✓

        windows = windows + self.mlp(self.norm2(windows))  # second sublayer unchanged

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
                 num_heads=4,          # FIX 4: expose num_heads (was hardcoded inside SwinBlock)
                 norm='linear'):
        super().__init__()

        self.norm = norm
        self.n_query_channels = n_query_channels
        self.embedding_dim = embedding_dim

        self.conv_embed = nn.Conv2d(in_channels, embedding_dim, 3, 1, 1)

        self.swin = nn.Sequential(
            SwinBlock(embedding_dim, num_heads=num_heads),
            SwinBlock(embedding_dim, num_heads=num_heads)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, dim_out)
        )

        # FIX 2: Remove query_conv (1x1 conv that projected to n_query_channels
        # channels and then sliced spatial positions as pseudo-queries).
        # We now use swin_features directly as queries — their last dim is
        # embedding_dim, which is what PixelWiseDotProduct requires for the
        # dot product with swin_features (also embedding_dim channels).
        # This is semantically equivalent to how mViT uses its transformer
        # output tokens as queries.

        self.dot_product_layer = PixelWiseDotProduct()

    def forward(self, x):

        B, C, H, W = x.shape

        x_embed = self.conv_embed(x)
        swin_features = self.swin(x_embed)   # (B, embedding_dim, H, W)

        # FIX 3: Compute range_attention_maps BEFORE the norm branching block.
        # Previously this was after the block, so the softmax branch returned
        # None for range_attention_maps, causing a crash in UnetAdaptiveBins.
        #
        # FIX 2 (continued): Query shape is now (B, n_query_channels, embedding_dim).
        # swin_features: (B, embedding_dim, H, W)
        # → flatten spatial:  (B, embedding_dim, H*W)
        # → permute:          (B, H*W, embedding_dim)
        # → slice first n_query_channels rows: (B, n_query_channels, embedding_dim)
        #
        # PixelWiseDotProduct asserts  c == ck  i.e.  embedding_dim == embedding_dim ✓
        # regardless of whether n_query_channels equals embedding_dim or not.
        queries = swin_features.flatten(2).permute(0, 2, 1)        # (B, H*W, embedding_dim)
        queries = queries[:, :self.n_query_channels, :]             # (B, n_query_channels, embedding_dim)
        range_attention_maps = self.dot_product_layer(swin_features, queries)  # (B, n_query_channels, H, W)

        # ---- bin width regression ----
        pooled = self.global_pool(swin_features).view(B, -1)   # (B, embedding_dim)
        y = self.regressor(pooled)                              # (B, dim_out)

        if self.norm == 'linear':
            y = torch.relu(y) + 0.1
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps   # range_attention_maps now valid ✓
        else:
            y = torch.sigmoid(y)

        y = y / y.sum(dim=1, keepdim=True)

        return y, range_attention_maps
