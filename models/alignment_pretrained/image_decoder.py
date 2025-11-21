import math
import torch
from torch import nn
from einops import rearrange

# your Transformer block
from .vit import Transformer

# ---------------- helpers ----------------
def _infer_grid(n_patches: int):
    g = int(math.sqrt(n_patches))
    if g * g != n_patches:
        raise ValueError(f"Cannot infer square grid from N={n_patches}. "
                         "Pass grid_hw=(H_patches, W_patches) explicitly.")
    return g, g

@torch.no_grad()
def unpatchify_pixel(
    pred_pixels: torch.Tensor,   # (B, N, 3) one RGB per patch
    patch_size: int = 16,
    grid_hw: tuple[int, int] | None = None
) -> torch.Tensor:
    """
    Tile each predicted pixel to its P×P patch and reassemble to an image.
    Returns (B, 3, H, W), where H = H_patches*P and W = W_patches*P.
    """
    B, N, C = pred_pixels.shape
    assert C == 3, f"Expected RGB last dim=3, got {C}"
    H_p, W_p = _infer_grid(N) if grid_hw is None else grid_hw
    assert H_p * W_p == N, f"grid_hw={grid_hw} does not match N={N}"
    P = patch_size

    patches = pred_pixels[..., None, None].expand(-1, -1, -1, P, P)   # (B, N, 3, P, P)
    img = rearrange(patches, "b (hp wp) c ph pw -> b c (hp ph) (wp pw)", hp=H_p, wp=W_p)
    return img

# --------------- main decoder ---------------
class MAEDecoder(nn.Module):
    """
    No masking. Predicts a single pixel (RGB=3) for **all** patches.

    Args:
        num_patches: total # of patch tokens (e.g., 14*14=196 for 224x224 @ 16x16)
        encoder_dim: channel dim of incoming tokens
        decoder_dim: internal decoder token dim
        pixel_dim:   output channels per patch (default 3 for RGB)
        pos_embed_weight: optional tensor to initialize positional embedding,
                          shape (num_patches, decoder_dim)
        freeze_pos_emb: if True, positional embedding is frozen
    """
    def __init__(
        self,
        *,
        num_patches: int,
        encoder_dim: int,
        decoder_dim: int,
        decoder_depth: int = 1,
        decoder_heads: int = 8,
        decoder_dim_head: int = 64,
        pixel_dim: int = 3,
        pos_embed_weight: torch.Tensor | None = None,
        freeze_pos_emb: bool = False,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        # project encoder tokens to decoder dim (identity if same)
        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim else nn.Identity()
        )

        # decoder and positional embedding (learned or user-initialized)
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4,
        )
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        if pos_embed_weight is not None:
            if pos_embed_weight.shape != (num_patches, decoder_dim):
                raise ValueError(
                    f"pos_embed_weight must be (num_patches, decoder_dim) = "
                    f"({num_patches}, {decoder_dim}), got {tuple(pos_embed_weight.shape)}"
                )
            with torch.no_grad():
                self.decoder_pos_emb.weight.copy_(pos_embed_weight)
        if freeze_pos_emb:
            self.decoder_pos_emb.weight.requires_grad_(False)

        # final head -> per-patch pixel (RGB by default)
        self.to_pixel = nn.Linear(decoder_dim, pixel_dim)

    def forward(
        self,
        encoded_tokens: torch.Tensor,   # (B, N, C_enc)  ALL patch tokens, no CLS
    ) -> torch.Tensor:
        B, N, Cenc = encoded_tokens.shape
        if N != self.num_patches:
            raise ValueError(f"encoded_tokens has {N} patches, expected {self.num_patches}")
        if Cenc != self.encoder_dim:
            raise ValueError(f"encoded_tokens dim {Cenc}, expected encoder_dim={self.encoder_dim}")

        device = encoded_tokens.device
        # project to decoder space
        x = self.enc_to_dec(encoded_tokens)                         # (B, N, D)

        # add positional embedding to **all** tokens (no mask)
        pos_ids = torch.arange(self.num_patches, device=device)     # (N,)
        pos = self.decoder_pos_emb(pos_ids).unsqueeze(0)            # (1, N, D)
        x = x + pos                                                 # (B, N, D)

        # transformer
        x = self.decoder(x)                                         # (B, N, D)

        # per-patch pixel prediction
        pred_pixels = self.to_pixel(x)                               # (B, N, 3)
        img = unpatchify_pixel(pred_pixels, patch_size=16, grid_hw=(14, 14)) 
        return pred_pixels, img
