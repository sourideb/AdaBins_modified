"""
swinv1.py
---------
Drop-in replacement for miniSwin.py.

Adds the core Swin Transformer mechanics that were missing:
  1. Relative position bias  (Eq.4 of the Swin paper)
  2. Cyclic shift            (efficient SW-MSA)
  3. Attention masking       (prevents cross-region attention after cyclic shift)
  4. Alternating W-MSA / SW-MSA blocks (every pair of blocks)
  5. Proper pre-norm architecture  (LN before each sub-layer)

Public interface is identical to miniSwin.SwinBins so only the import
line in unet_adaptive_bins.py needs to change:
    from .swinv1 import SwinBins
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PixelWiseDotProduct


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def window_partition(x, window_size):
    """
    Split a (B, H, W, C) feature map into non-overlapping windows.

    Args:
        x          : (B, H, W, C)  — H, W must be divisible by window_size
        window_size: int

    Returns:
        windows: (B * nW, window_size, window_size, C)
                 where nW = (H/ws) * (W/ws)
    """
    B, H, W, C = x.shape
    x = x.view(B,
               H // window_size, window_size,
               W // window_size, window_size,
               C)
    # (B, nH, ws, nW, ws, C) → (B, nH, nW, ws, ws, C) → (B*nH*nW, ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reconstruct feature map from windows.

    Args:
        windows    : (B * nW, window_size, window_size, C)
        window_size: int
        H, W       : ints — target spatial dimensions

    Returns:
        x: (B, H, W, C)
    """
    nW = (H // window_size) * (W // window_size)
    B  = windows.shape[0] // nW
    x  = windows.view(B,
                      H // window_size,
                      W // window_size,
                      window_size,
                      window_size,
                      -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ---------------------------------------------------------------------------
# Window Attention  (W-MSA and SW-MSA share this module)
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention with relative position bias.

    When called with mask=None  → W-MSA  (regular windows)
    When called with mask!=None → SW-MSA (shifted windows; mask prevents
                                          attention across cyclic-shift seams)

    Relative position bias:
        Attention(Q,K,V) = Softmax(Q K^T / sqrt(d)  +  B) V
    where B is indexed from a learnable table of size
    (2*window_size-1) x (2*window_size-1) per head.
    """

    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size          # Wh == Ww (square windows)
        self.num_heads   = num_heads
        self.scale       = (dim // num_heads) ** -0.5

        # ---- Relative position bias table: (2Wh-1)*(2Ww-1), nH ----
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Pre-compute and register relative position indices (constant)
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        # torch.meshgrid default indexing is 'ij' — safe across PyTorch versions
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flat = torch.flatten(coords, 1)                         # 2, Wh*Ww

        # pairwise relative coordinates
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]        # 2, N, N
        rel = rel.permute(1, 2, 0).contiguous()                        # N, N, 2
        rel[:, :, 0] += self.window_size - 1                           # shift ≥ 0
        rel[:, :, 1] += self.window_size - 1
        rel[:, :, 0] *= 2 * self.window_size - 1                       # row-major index
        relative_position_index = rel.sum(-1)                          # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv      = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x   : (B_win, N, C)   where B_win = B * nW,  N = ws^2
            mask: (nW, N, N) or None
        Returns:
            x   : (B_win, N, C)
        """
        B_, N, C = x.shape
        head_dim = C // self.num_heads

        # QKV projection → split heads
        qkv = (self.qkv(x)
               .reshape(B_, N, 3, self.num_heads, head_dim)
               .permute(2, 0, 3, 1, 4))              # 3, B_, nH, N, hd
        q, k, v = qkv.unbind(0)                      # each: (B_, nH, N, hd)

        # Scaled dot-product
        attn = (q * self.scale) @ k.transpose(-2, -1)   # (B_, nH, N, N)

        # Add relative position bias
        rel_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, self.num_heads)                     # N, N, nH
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()  # nH, N, N
        attn = attn + rel_bias.unsqueeze(0)

        # Apply SW-MSA mask  (large negative → zeroed after softmax)
        if mask is not None:
            nW = mask.shape[0]
            # reshape to (B, nW, nH, N, N) so mask broadcasts over B and nH
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ---------------------------------------------------------------------------
# Swin Transformer Block  (one W-MSA or SW-MSA step)
# ---------------------------------------------------------------------------

class SwinTransformerBlock(nn.Module):
    """
    A single Swin Transformer block.

    shift_size = 0           → W-MSA  (regular windows)
    shift_size = window_size // 2 → SW-MSA (shifted windows via cyclic roll)

    Pre-norm (LN before each sub-layer), residual after:
        z_hat  =  (S)W-MSA( LN(z_prev) )  +  z_prev
        z      =  MLP( LN(z_hat) )         +  z_hat
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        assert 0 <= shift_size < window_size, \
            f"shift_size ({shift_size}) must be in [0, window_size={window_size})"

        self.window_size = window_size
        self.shift_size  = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, attn_mask):
        """
        Args:
            x        : (B, H, W, C)
            attn_mask: (nW, ws^2, ws^2) for SW-MSA; None for W-MSA

        Returns:
            x: (B, H, W, C)
        """
        B, H, W, C = x.shape

        # ── Attention sub-layer ──────────────────────────────────────────
        shortcut = x
        x = self.norm1(x)

        # Pad so H, W are divisible by window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))   # (B, Hp, Wp, C)
        Hp, Wp = x.shape[1], x.shape[2]

        # Cyclic shift  (SW-MSA only)
        if self.shift_size > 0:
            x = torch.roll(x,
                           shifts=(-self.shift_size, -self.shift_size),
                           dims=(1, 2))

        # Partition → attention → merge
        x_wins = window_partition(x, self.window_size)            # (B*nW, ws, ws, C)
        x_wins = x_wins.view(-1, self.window_size ** 2, C)        # (B*nW, ws^2, C)
        x_wins = self.attn(x_wins, mask=attn_mask)                # (B*nW, ws^2, C)
        x_wins = x_wins.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x_wins, self.window_size, Hp, Wp)      # (B, Hp, Wp, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))

        # Remove padding and first residual
        x = x[:, :H, :W, :].contiguous()
        x = shortcut + x

        # ── MLP sub-layer ────────────────────────────────────────────────
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# SwinBins  — same interface as miniSwin.SwinBins
# ---------------------------------------------------------------------------

class SwinBins(nn.Module):
    """
    Swin-based adaptive-bin-width estimator for AdaBins.

    This is a drop-in replacement for miniSwin.SwinBins with proper
    Swin Transformer mechanics (shifted windows, cyclic shift, relative
    position bias, attention masking).

    Input / output contract is identical to the original:
        forward(x) → (bin_widths_normed, range_attention_maps)

    Args:
        in_channels    : channels of the incoming feature map (128 from decoder)
        n_query_channels: channels for range-attention queries  (default 128)
        dim_out        : number of depth bins                   (= n_bins)
        embedding_dim  : internal Swin feature dimension        (default 128)
        norm           : bin-width normalisation — 'linear' | 'softmax' | 'sigmoid'
        window_size    : self-attention window size             (default 7)
        num_heads      : number of attention heads              (default 4)

    NOTE: PixelWiseDotProduct requires  embedding_dim == n_query_channels
          (both default to 128, matching the AdaBins decoder output).
    """

    def __init__(self,
                 in_channels,
                 n_query_channels=128,
                 dim_out=256,
                 embedding_dim=128,
                 norm='linear',
                 window_size=7,
                 num_heads=4):
        super().__init__()

        self.norm             = norm
        self.n_query_channels = n_query_channels
        self.window_size      = window_size
        self.shift_size       = window_size // 2
        self.embedding_dim    = embedding_dim

        # ── Input projection ────────────────────────────────────────────
        self.conv_embed = nn.Conv2d(in_channels, embedding_dim,
                                    kernel_size=3, padding=1)

        # ── Two Swin blocks: W-MSA then SW-MSA ──────────────────────────
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                embedding_dim, num_heads,
                window_size=window_size,
                shift_size=0                  # W-MSA
            ),
            SwinTransformerBlock(
                embedding_dim, num_heads,
                window_size=window_size,
                shift_size=window_size // 2   # SW-MSA
            ),
        ])
        self.norm_layer = nn.LayerNorm(embedding_dim)

        # ── Bin-width regression (global branch) ────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, dim_out),
        )

        # ── Range-Attention Maps (pixel-wise branch) ─────────────────────
        self.query_conv       = nn.Conv2d(embedding_dim, n_query_channels, kernel_size=1)
        self.dot_product_layer = PixelWiseDotProduct()

    # -----------------------------------------------------------------------
    def _compute_attn_mask(self, H, W, device):
        """
        Build the attention mask required by the SW-MSA block.

        After cyclic shift the feature map is divided into 9 regions
        (3 horizontal × 3 vertical slabs). Tokens from different regions
        that end up in the same window must NOT attend to each other.
        We assign each region a unique integer label, partition into
        windows, then mask pairs whose labels differ with -100 (→ ~0
        after softmax).

        The mask is computed on-the-fly and is cheap relative to the
        attention itself. Cache it externally if profiling shows it matters.

        Returns:
            attn_mask: (nW, ws^2, ws^2)  on `device`
        """
        ws  = self.window_size
        sft = self.shift_size

        # Pad to multiples of window_size (same logic as SwinTransformerBlock)
        Hp = int(np.ceil(H / ws)) * ws
        Wp = int(np.ceil(W / ws)) * ws

        # Label map for the 9 regions
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (slice(0, -ws), slice(-ws, -sft), slice(-sft, None))
        w_slices = (slice(0, -ws), slice(-ws, -sft), slice(-sft, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Partition into windows and compute pairwise label differences
        mask_windows = window_partition(img_mask, ws)       # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, ws * ws)       # (nW, ws^2)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, ws^2, ws^2)
        attn_mask = (attn_mask
                     .masked_fill(attn_mask != 0, -100.0)
                     .masked_fill(attn_mask == 0,   0.0))
        return attn_mask

    # -----------------------------------------------------------------------
    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            bin_widths_normed  : (B, dim_out)       — sums to 1 per sample
            range_attention_maps: (B, n_query_channels, H, W)
        """
        B, C, H, W = x.shape

        # ── Embed and convert to (B, H, W, C) for Swin ops ─────────────
        feat = self.conv_embed(x)               # (B, embedding_dim, H, W)
        feat = feat.permute(0, 2, 3, 1)         # (B, H, W, embedding_dim)

        # ── Attention mask for SW-MSA ────────────────────────────────────
        attn_mask = self._compute_attn_mask(H, W, x.device)

        # ── Block 0: W-MSA  (no mask) ────────────────────────────────────
        feat = self.swin_blocks[0](feat, attn_mask=None)

        # ── Block 1: SW-MSA (with cyclic-shift mask) ─────────────────────
        feat = self.swin_blocks[1](feat, attn_mask=attn_mask)

        feat = self.norm_layer(feat)
        swin_out = feat.permute(0, 3, 1, 2)     # (B, embedding_dim, H, W)

        # ── Bin-width regression ─────────────────────────────────────────
        pooled = self.global_pool(swin_out).view(B, -1)   # (B, embedding_dim)
        y = self.regressor(pooled)                         # (B, dim_out)

        if self.norm == 'linear':
            y = torch.relu(y) + 0.1
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), None
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)                # normalise to sum=1

        # ── Range-Attention Maps ─────────────────────────────────────────
        # query_conv maps embedding_dim → n_query_channels (both 128 by default)
        queries = self.query_conv(swin_out)                 # (B, n_query_channels, H, W)
        queries = queries.flatten(2).permute(0, 2, 1)       # (B, H*W, n_query_channels)

        # PixelWiseDotProduct needs K: (B, cout, ck) where ck == embedding_dim
        # queries[:, :n_query_channels, :] → (B, n_query_channels, n_query_channels)
        # This works because n_query_channels == embedding_dim == 128 by default.
        range_attention_maps = self.dot_product_layer(
            swin_out,
            queries[:, :self.n_query_channels, :]
        )                                                   # (B, n_query_channels, H, W)

        return y, range_attention_maps
