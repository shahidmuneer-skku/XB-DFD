import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# small helpers
# --------------------------
def _infer_grid(n_patches: int):
    g = int(math.sqrt(n_patches))
    if g * g != n_patches:
        raise ValueError(f"Cannot infer square grid from N={n_patches}. "
                         f"Got N={n_patches}. Pass grid_hw=(H_patches, W_patches) explicitly.")
    return g, g

def _gn(ch):
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)

class DoubleConv(nn.Module):
    """Conv → GN → SiLU → Conv → GN → SiLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _gn(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _gn(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    """Downscale by 2 with stride-2 conv, then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            _gn(out_ch),
            nn.SiLU(inplace=True),
        )
        self.block = DoubleConv(out_ch, out_ch)

    def forward(self, x):
        x = self.down(x)
        return self.block(x)
class Up(nn.Module):
    """Upscale to the skip's size + Conv, then concat skip, then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.block  = DoubleConv(out_ch * 2, out_ch)  # concat with skip (same out_ch)

    def forward(self, x, skip):
        # resize feature map exactly to the skip connection's H×W
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

        
class UNetImageDecoder(nn.Module):
    """
    From [B, N, D] tokens -> [B, C_out, H_img, W_img].
    Defaults for ViT/MAE-style patches: N=196 (14x14), image 224x224 → patch_size=16.
    """
    def __init__(
        self,
        num_patches: int = 49,
        token_dim: int = 512,
        out_channels: int = 1,       # set 1 for mask, 3 for RGB
        base_channels: int = 256,    # width of the UNet
        img_size: int = 224,
        grid_hw: tuple[int, int] | None = None,
    ):
        super().__init__()
        if grid_hw is None:
            H_p, W_p = _infer_grid(num_patches)
        else:
            H_p, W_p = grid_hw
            assert H_p * W_p == num_patches, "grid_hw must match num_patches"

        assert img_size % H_p == 0 and img_size % W_p == 0, \
            f"img_size {img_size} not divisible by token grid {(H_p, W_p)}"

        self.H_p, self.W_p = H_p, W_p
        self.img_h, self.img_w = img_size, img_size
        # total upscale factors (usually 224/14 = 16 in both dims)
        self.up_s_h = self.img_h // H_p
        self.up_s_w = self.img_w // W_p
        assert self.up_s_h == self.up_s_w, "non-square upscales not supported (got H and W different)."
        self.total_up = int(self.up_s_h)  # e.g., 16
        assert self.total_up & (self.total_up - 1) == 0, \
            "total upscale should be a power of two for clean x2 steps (e.g., 16 = 2^4)."

        # 1×1 to go from token_dim to base feature channels on 14×14 grid
        self.token_proj = nn.Conv2d(token_dim, base_channels, kernel_size=1)

        # -------- UNet encoder (14→7→4→2→1) --------
        # 14x14
        self.inc   = DoubleConv(base_channels, base_channels)
        # 7x7
        self.down1 = Down(base_channels, base_channels * 2)
        # 4x4
        self.down2 = Down(base_channels * 2, base_channels * 4)
        # 2x2
        self.down3 = Down(base_channels * 4, base_channels * 8)
        # 1x1 (bottleneck)
        self.down4 = Down(base_channels * 8, base_channels * 8)

        # -------- UNet decoder back to 14x14 --------
        self.up1 = Up(base_channels * 8, base_channels * 8)   # 1→2, skip=down3
        self.up2 = Up(base_channels * 8, base_channels * 4)   # 2→4, skip=down2
        self.up3 = Up(base_channels * 4, base_channels * 2)   # 4→7, skip=down1
        self.up4 = Up(base_channels * 2, base_channels)       # 7→14, skip=inc

        # -------- Super-resolution head: 14→224 (×16 via 4 ×2 steps) --------
        sr_blocks = []
        ch = base_channels
        steps = int(math.log2(self.total_up))   # e.g., 4 for ×16
        for _ in range(steps):
            sr_blocks += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                DoubleConv(ch, max(ch // 2, 64)),
            ]
            ch = max(ch // 2, 64)
        self.superres = nn.Sequential(*sr_blocks)

        # final 1×1 to get desired channels; output is raw logits
        self.out = nn.Conv2d(ch, out_channels, kernel_size=1)

        # (Optional) lightweight head to produce a per-patch embedding again (for your API)
        # Here we just pool features at 14×14 back into [B, N, token_dim] if you want something non-None.
        self.patch_head = nn.Sequential(
            nn.Conv2d(base_channels, token_dim, kernel_size=1),
            _gn(token_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, tokens: torch.Tensor):
        """
        tokens: [B, N, D]
        returns:
            patch_embed: [B, N, D]  (lightweight re-embedding of the 14×14 features)
            img:         [B, C_out, H_img, W_img]  (raw logits; apply sigmoid if mask)
        """
        B, N, D = tokens.shape
        assert N == self.H_p * self.W_p, f"tokens N={N} ≠ grid {self.H_p}×{self.W_p}"

        # [B, N, D] → [B, D, H_p, W_p]
        x = tokens.view(B, self.H_p, self.W_p, D).permute(0, 3, 1, 2).contiguous()
        x = self.token_proj(x)

        # encoder
        x1 = self.inc(x)            # 14×14,   ch=base
        x2 = self.down1(x1)         # 7×7,    ch=2*base
        x3 = self.down2(x2)         # 4×4,    ch=4*base
        x4 = self.down3(x3)         # 2×2,    ch=8*base
        x5 = self.down4(x4)         # 1×1,    ch=8*base

        # decoder back to 14×14
        y = self.up1(x5, x4)        # 2×2
        y = self.up2(y,  x3)        # 4×4
        y = self.up3(y,  x2)        # 7×7
        y = self.up4(y,  x1)        # 14×14   (ch=base_channels)

        # optional per-patch embedding head to keep your original return signature
        patch_feat = self.patch_head(y)                    # [B, D, 14, 14]
        patch_embed = patch_feat.permute(0, 2, 3, 1).contiguous().view(B, -1, patch_feat.size(1))  # [B, N, D]

        # super-res to full image res
        z = self.superres(y)                               # [B, *, H_img, W_img]
        img_logits = self.out(z)                           # [B, out_ch, H_img, W_img]
        return patch_embed, img_logits
