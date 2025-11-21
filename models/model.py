from __future__ import annotations

"""Optimised multimodal moderator.

* graceful **None** handling for any missing modality.
* image‑to‑tokens defined once.
* losses are computed **only** for modalities that are present; missing branches contribute 0.
* removed redundant conversions and Debug prints.
* MultiheadAttention flattened bug fixed.
"""

import math
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.masked_encoder import MaskEncoder as MaskEncoder, TokenTypes, Attention
import lpips

# def token_dropout(tokens: torch.Tensor, p: float = 0.15, training: bool = True) -> torch.Tensor:
#     """
#     Randomly **zeros out entire patch / token vectors** with probability *p*.

#     Parameters
#     ----------
#     tokens : Tensor
#         Input tensor of shape **[B, N, D]** (batch, #tokens, embeddim).
#     p : float, default 0.15
#         Drop probability per token (0≤p<1).  
#         Set to 0.0 to disable.
#     training : bool, default True
#         If False (e.g. during evaluation), the function is a noop.

#     Returns
#     -------
#     Tensor
#         Tensor of same shape as *tokens* where a subset of tokens have been set to 0.
#     """
#     if (not training) or p <= 0.0:
#         return tokens

#     # build dropout mask: True where we KEEP the token
#     keep_mask = torch.rand(tokens.shape[:2], device=tokens.device) >= p        # [B, N]
#     keep_mask = keep_mask.unsqueeze(-1).type_as(tokens)                        # [B, N, 1]

#     # scale to keep expected magnitude (as in standard dropout)
#     scale = 1.0 / (1.0 - p)
#     return tokens * keep_mask * scale

# def block_token_dropout(
#     tokens: torch.Tensor,
#     p: float = 0.15,
#     block_size: int | None = None,
#     training: bool = True
# ) -> torch.Tensor:
#     """
#     Randomly zero out contiguous spans ("blocks") of tokens with overall drop
#     probability ~p.  If block_size is None, we choose it so that blocks of size
#     block_size^2 cover roughly p fraction of tokens.
    
#     tokens: [B, N, D]
#     p: total fraction of tokens to drop across the sequence
#     block_size: length of each contiguous block (in tokens).  If None, we pick
#                 block_size = max(1, int(sqrt(N * p)))
#     """
#     # if (not training) or p <= 0.0:
#     #     return tokens

#     B, N, D = tokens.shape

#     # choose block_size if not given
#     if block_size is None:
#         # want block_size^2 / N ≈ p  ⇒ block_size ≈ sqrt(N * p)
#         block_size = max(1, int(math.sqrt(N * p)))
#     # how many blocks per sample?
#     # we approximate: num_blocks * block_size / N ≈ p  ⇒  num_blocks ≈ p * N / block_size
#     num_blocks = max(1, int(p * N / block_size))

#     # start with all-ones mask
#     mask = torch.ones((B, N), device=tokens.device, dtype=torch.bool)

#     for b in range(B):
#         for _ in range(num_blocks):
#             start = torch.randint(0, N - block_size + 1, (1,)).item()
#             mask[b, start:start + block_size] = False

#     # expand to [B, N, 1]
#     mask = mask.unsqueeze(-1)
#     # scale so that E[out] == in
#     keep_prob = mask.float().mean(dim=1, keepdim=True)  # [B,1,1]
#     scale = 1.0 / keep_prob.clamp(min=1e-6)

#     return tokens * mask.type_as(tokens) * scale



# def shuffle(tok):
#     # if not self.training: return tok
#     idx = torch.randperm(tok.size(1), device=tok.device)
#     return tok[:, idx, :]

# def make_views(tok):
#     # v1 = token_dropout(tok, p=0.15)
#     # v1 = token_dropout(tok, p=0.15)
#     # v2 = token_dropout(shuffle(tok), p=0.15)
#     v1 = block_token_dropout(tok, p=0.3)
#     v2 = block_token_dropout(shuffle(tok), p=0.3)
#     return v1, v2
# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def feature_dropout(x: torch.Tensor, p: float = 0.3) -> torch.Tensor:
    """
    x: [B, N, D] or [B, D, N] — must be clear which one you're using!
    Drops a subset of D dimensions and scales the rest.
    """
   

    B, D, N = x.shape  # assuming [B, D, N]

    # Create dropout mask for features (dim D)
    keep_mask = torch.rand(D, device=x.device) >= p   # [D]
    scale = 1.0 / keep_mask.float().mean().clamp(min=1e-6)

    x = x * keep_mask[:, None] * scale  # [D, 1] broadcasted over B and N
    return x

def block_token_dropout(
    tokens: torch.Tensor,
    grid_shape: Tuple[int, ...],  # e.g., (T, H, W) or (H, W)
    p: float = 0.15,
    block_size: int | None = None,
    training: bool = True
) -> torch.Tensor:
    """
    Block token dropout that respects spatial/temporal layout.
    
    Parameters
    ----------
    tokens: [B, N, D]  - flattened patch embeddings
    grid_shape: Tuple  - original patch grid shape: (T,H,W) or (H,W)
    p: float           - total fraction to drop
    block_size: int    - side length of square (2D) or cube (3D) block
    training: bool     - toggle for training vs eval
    """
    if not training or p <= 0.0:
        return tokens

    B, N, D = tokens.shape
    dims = len(grid_shape)
    assert math.prod(grid_shape) == N, f"grid_shape {grid_shape} does not match tokens N={N}"

    # Reshape tokens into [B, *grid, D]
    tokens_reshaped = tokens.view(B, *grid_shape, D)

    # Create mask
    mask = torch.ones((B, *grid_shape), dtype=torch.bool, device=tokens.device)

    # Estimate number of blocks per sample
    block_size = block_size or max(1, int((N * p) ** (1 / dims)))
    total_blocks = max(1, int(p * N / (block_size ** dims)))

    for b in range(B):
        for _ in range(total_blocks):
            # Sample top-left corner
            starts = [torch.randint(0, max(1, s - block_size + 1), (1,)).item() for s in grid_shape]
            slices = tuple(slice(s, s + block_size) for s in starts)
            mask[b][slices] = False

    # Expand mask to match embedding dimension
    mask = mask.unsqueeze(-1)  # shape: [B, T, H, W, 1] or [B, H, W, 1]
    scale = 1.0 / mask.float().mean(dim=tuple(range(1, dims + 1)), keepdim=True).clamp(min=1e-6)

    tokens_dropped = tokens_reshaped * mask * scale
    return tokens_dropped.view(B, N, D)

def token_dropout(tokens: torch.Tensor, p: float = 0.15, training: bool = True) -> torch.Tensor:
    if (not training) or p <= 0.0:
        return tokens
    # always drop entire feature-channels
    if tokens.dim() == 2:
        # [B, D]
        keep = (torch.rand(tokens.shape, device=tokens.device) >= p).float()
    elif tokens.dim() == 3:
        # [B, N, D]  → same mask across N
        B, N, D = tokens.shape
        keep = (torch.rand(B, 1, D, device=tokens.device) >= p).float()
    else:
        raise ValueError(f"Cannot token-dropout a {tokens.dim()}-D tensor")

    return tokens * keep / (1 - p)


def shuffle_sequence(tok: torch.Tensor) -> torch.Tensor:
    """
    Shuffle either:
      – the token axis of a [B, N, D] tensor, or
      – the batch axis of a [B, D] tensor.
    """
    if tok.dim() == 3:
        # [B, N, D] → shuffle along the N (token) dimension
        idx = torch.randperm(tok.size(1), device=tok.device)
        return tok[:, idx, :]

    elif tok.dim() == 2:
        # [B, D] → shuffle samples in the batch
        idx = torch.randperm(tok.size(0), device=tok.device)
        return tok[idx, :]

    else:
        raise ValueError(f"shuffle_sequence only supports 2D or 3D tensors, got shape {tuple(tok.shape)}")


def random_masking(x: torch.Tensor, mask_ratio: float = 0.3, mask_token: Optional[torch.Tensor] = None):
    """
    Args:
        x:          Input tensor of shape [B, N] or [B, N, D]
        mask_ratio: Fraction of tokens to mask (e.g. 0.3 means keep 70%)
        mask_token: Optional tensor of shape [D] to use as replacement
    
    Returns:
        x_masked:   Masked tensor [B, N]
        mask:       Boolean mask [B, N] — True where masked
        ids_restore:Permutation index to restore original order
    """
    assert x.dim() == 2, "Expected [B, N], got {}".format(x.shape)
    B, N = x.shape
    len_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)  # [B, N]
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]  # [B, len_keep]

    # Gather kept tokens
    x_keep = torch.gather(x, dim=1, index=ids_keep)  # [B, len_keep]

    # Prepare mask tokens
    if mask_token is None:
        mask_token = torch.zeros(1, device=x.device)  # [1]

    mask_tokens = mask_token.expand(B, N - len_keep)  # [B, N-len_keep]

    x_merged = torch.cat([x_keep, mask_tokens], dim=1)  # [B, N]
    x_masked = torch.gather(x_merged, dim=1, index=ids_restore)  # [B, N]

    # Mask: True where token was replaced
    mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
    mask[:, :len_keep] = False
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

def make_views(tok: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # tok must be [B, N, D]
    v1 = random_masking(tok, mask_ratio=0.1)
    # v2 = shuffle_sequence(random_masking(tok, mask_ratio=0.5))
    return v1



    
def present(x: Optional[torch.Tensor]) -> bool:
    return (x is not None) and (x.numel() > 0)


def zero_tensor_like(ref: torch.Tensor, *shape) -> torch.Tensor:
    """Utility to create an all‑zeros tensor sharing dtype/device with *ref*."""
    return ref.new_zeros(*shape)


# -----------------------------------------------------------------------------
# Simple contrastive‑loss wrapper that tolerates empty inputs
# -----------------------------------------------------------------------------

# class PatchContrastiveLoss(nn.Module):
#     """NT-Xent loss over patch embeddings.
#        Expects x and y of shape [B, N, D] with N_x == N_y == N_patches."""
#     def __init__(self, temperature: float = 0.1):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         if x.numel() == 0 or y.numel() == 0:
#             return x.new_tensor(0.0)  # missing modality ⇒ 0 loss

#         B, N, D = x.shape
#         x = F.normalize(x, dim=-1).view(B * N, D)
#         y = F.normalize(y, dim=-1).view(B * N, D)
#         logits = torch.matmul(x, y.t()) / self.temperature
#         labels = torch.arange(B * N, device=x.device)
#         return F.cross_entropy(logits, labels)

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.norm = nn.LayerNorm(dim)
#         self.attend = nn.Softmax(dim = -1)
#         self.dropout = nn.Dropout(dropout)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         x = self.norm(x)
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)
        

class MaskedAutoencoder(nn.Module):
    """
    MAE-style masked autoencoder for patch embeddings.
    You give it input tokens [B, N, D], it:
      1. Randomly masks a fraction of them,
      2. Encodes the masked sequence with your encoder,
      3. Decodes back to D-dim embeddings,
      4. Computes MSE only on the masked positions.
    """
    def __init__(
        self,
        encoder: nn.Module,
        decoder_dim: int,
        mask_ratio: float = 0.75
    ):
        """
        encoder      – nn.Module that takes tokens [B,N,D] → some latent [B,N,D]
        decoder_dim  – width of hidden layer in the decoder
        mask_ratio   – fraction of tokens to mask out
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        D = 512
        self.encoder = nn.Sequential(
            nn.Linear(D, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
        )
        # # one mask token (learnable) for all positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))

        # simple 2-layer decoder: D→decoder_dim→D
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(),
            nn.Linear(256, D),
        )

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking.
        Returns:
          x_masked   [B, N, D]    – with masked positions replaced by mask_token
          mask       [B, N]       – bool mask (True = was masked)
          ids_restore [B, N]      – to invert the shuffling
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        # 1. generate noise for each sample, then sort to get shuffle indices
        noise = torch.rand(B, N, device=x.device)  # uniform [0,1)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 2. keep the first len_keep patches, mask the rest
        ids_keep = ids_shuffle[:, :len_keep]
        x_keep = torch.gather(
            x, 1,
            ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, len_keep, D]

        # 3. prepare a full [B, N, D] with x_keep + mask_tokens
        mask_tokens = self.mask_token.expand(B, N - len_keep, D)
        x_merged = torch.cat([x_keep, mask_tokens], dim=1)  # [B, N, D]
        # 4. unshuffle to restore original ordering
        x_masked = torch.gather(
            x_merged, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, N, D]

        # final boolean mask of which were masked
        mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        mask[:, :len_keep] = False
        mask = torch.gather(mask, 1, ids_restore)  # align with original positions

        return x_masked, mask, ids_restore

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens : [B, N, D] patch embeddings (vision or audio)
        returns  : scalar reconstruction loss (MSE on masked patches)
        """
        # 1. mask
        x_masked, mask, _ = self.random_masking(tokens)

        # 2. encode masked sequence
        #    assume your encoder can take raw tokens → contextualized tokens
        z = self.encoder(x_masked)       # [B, N, D]

        # 3. decode all positions
        recon = self.decoder(z)          # [B, N, D]

        # 4. compute MSE only on masked positions
        loss = F.mse_loss(
            recon[mask],     # predicted at masked positions
            tokens[mask],    # ground-truth embeddings
            reduction="mean"
        )
        return loss



class PatchContrastiveLoss(nn.Module):
    """NT-Xent loss over patch embeddings.
       Expects x and y of shape [B, N, D] with N_x == N_y == N_patches."""
    def __init__(self, temperature: float = 0.8):
        super().__init__()
        self.temperature = temperature
        # D = 512
        # self.x_predictor= nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        D = 512
        self.encoder = nn.Sequential(
            nn.Linear(D, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        # one mask token (learnable) for all positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))

        # simple 2-layer decoder: D→decoder_dim→D
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, D),
        )
        # self.y_predictor= nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        B, N, D = x.shape
        x = F.normalize(x.contiguous()).view(B * N, D)
        y = F.normalize(y.contiguous()).view(B * N, D)
        
        z   =   self.encoder(x)
        pred =  self.decoder(z)
        # y=self.y_predictor(y)
        logits = torch.matmul(pred, y.t()) / self.temperature
        # pred_norm = x / x.norm(dim=1, keepdim=True)
        # y_norm = y / y.norm(dim=1, keepdim=True)
        # logits = (x_norm * y_norm).sum(dim=1) / self.temperature  # shape: (B*N,)
        
        # x = x.contiguous().view(B * N, D)
        # y = y.contiguous().view(B * N, D)
        # logits = F.cosine_similarity(x[None,:,:], y[:,None,:], dim=-1)
        # print(logits.shape)
        # Proper way to mask diagonals safely:
        diag_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
        logits.masked_fill_(diag_mask, float('-inf'))
        # logits.masked_fill_(diag_mask, -1e9)

        labels = torch.arange(logits.size(0), device=pred.device).long()
        # labels[0::2] += 1
        # labels[1::2] -= 1
        # print(labels.shape)

        return F.cross_entropy(logits, labels, reduction="mean").mean()# + 0.6 * F.mse_loss(pred, y)

    """NT-Xent loss over patch embeddings.
       Expects x and y of shape [B, N, D] with N_x == N_y == N_patches."""

# class TokenLoss(nn.Module):
#     def __init__(self, temperature: float = 0.7):
#         super().__init__()
#         self.temperature = temperature
#         # encoder/decoder as before…
#         D = 512
#         self.encoder = nn.Sequential(
#             nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, 64),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, D),
#         )

#         # self.hash = nn.Sequential(
#         #     nn.Linear(D,64),
#         #     nn.Tanh()
#         #     )

#         self.pretraining = False
#         # we’ll lazily create this once we see an input of dim D
#         self.mask_token: Optional[nn.Parameter] = None


#     def random_masking(self, x: torch.Tensor, mask_ratio: float = 0.3):
#         # now expect x: [B, N, D]
#         assert x.dim() == 3, "Expected [B, N, D]"
#         B, N, D = x.shape
#         len_keep = int(N * (1 - mask_ratio))

#         # 1) compute which tokens to keep
#         noise = torch.rand(B, N, device=x.device)
#         ids_shuffle  = torch.argsort(noise, dim=1)
#         ids_restore  = torch.argsort(ids_shuffle, dim=1)
#         ids_keep     = ids_shuffle[:, :len_keep]  # [B, len_keep]
#         x_keep       = torch.gather(
#             x, 1,
#             ids_keep.unsqueeze(-1).expand(-1, -1, D)
#         )  # [B, len_keep, D]

#         # 2) lazy-init mask_token at correct D
#         if self.mask_token is None or self.mask_token.shape[-1] != D:
#             # create as [D], on correct device
#             mt = torch.zeros(D, device=x.device)
#             self.mask_token = nn.Parameter(mt)
#             nn.init.trunc_normal_(self.mask_token, std=0.02)

#         # 3) build mask_tokens of shape [B, N_masked, D]
#         mask_tokens = (
#             self.mask_token
#                 .unsqueeze(0)      # [1, D]
#                 .unsqueeze(0)      # [1, 1, D]
#                 .expand(B, N - len_keep, D)
#         )

#         # 4) merge and unshuffle
#         x_merged = torch.cat([x_keep, mask_tokens], dim=1)  # [B, N, D]
#         x_masked = torch.gather(
#             x_merged, 1,
#             ids_restore.unsqueeze(-1).expand(-1, -1, D)
#         )  # [B, N, D]

#         # 5) boolean mask of masked positions
#         mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
#         mask[:, :len_keep] = False
#         mask = torch.gather(mask, 1, ids_restore)

#         return x_masked, mask, ids_restore

#     def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
#         # x: [B, N, D]
#         # if y is None:
#         #     x_masked, mask, _ = self.random_masking(x)
#         #     z    = self.encoder(x_masked)  # [B, N, latent]
#         #     pred = self.decoder(z)         # [B, N, D]
#         #     mse_loss = F.mse_loss(pred[mask], x[mask])
#         #     # return mse_loss

#         #     # y = F.normalize(x[mask], dim=-1)
#         #     # pred = F.normalize(pred[mask], dim=-1)
#         #     # logits = torch.matmul(x[mask], pred[mask].t()) / self.temperature
#         #     # diag_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
#         #     # logits.masked_fill_(diag_mask, float('-inf'))
#         #     # labels = torch.arange(logits.size(0), device=x.device).long()
#         #     # labels[0::2] += 1
#         #     # labels[1::2] -= 1
#         #     # return 0.1 * F.cross_entropy(logits, labels, reduction="mean").mean() + 0.9 * mse_loss
#         #     # x: [B, N, D]
            

#         #     # Normalize
#         #     # y = F.normalize(x_masked_tokens, dim=-1)
#         #     # pred_norm = F.normalize(pred_masked_tokens, dim=-1)
            
#         #     y = F.normalize(x[mask], dim=-1)
#         #     pred = F.normalize(pred[mask], dim=-1)
          
#         #     x = x.reshape(-1, x.size(-1))        # [total_tokens, D]
#         #     pred = pred.reshape(-1, pred.size(-1)) # [total_tokens, D]

#         #     y = torch.sign(self.hash(x))
#         #     pred = torch.sign(self.hash(pred))
#         #     # Compute logits [num_masked, num_masked]
#         #     logits = torch.matmul(y, pred.t()) / self.temperature

#         #     # diag_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
#         #     # logits.masked_fill_(diag_mask, float('-inf'))

#         #     labels = torch.arange(logits.size(0), device=logits.device).long()
#         #     # labels[0::2] += 1
#         #     # labels[1::2] -= 1

#         #     return 0.2 * F.cross_entropy(logits, labels, reduction="mean") + 0.8 * mse_loss

#         if y is None:
#             mask_ratio = np.random.uniform(0.25, 0.75)

#             x_masked, mask, _ = self.random_masking(x,mask_ratio=mask_ratio)
#             z = self.encoder(x_masked)
#             pred = self.decoder(z)
#             # Masked tokens only
#             # target = x[mask]    # [num_masked, D]
#             # predicted = pred[mask]
#             target = x[mask]    # [num_masked, D]
#             predicted = pred[mask]
#             mse_loss = F.mse_loss(predicted, target)
#             # Contrastive
#             # y_hash = torch.sign(self.hash(target))
#             # pred_hash = torch.sign(self.hash(predicted))
            
#             # target = F.normalize(target, dim=-1)
#             # predicted = F.normalize(predicted, dim=-1)
            
            
#             target = target.reshape(-1, target.size(-1))        # [total_tokens, D]
#             predicted = predicted.reshape(-1, predicted.size(-1)) # [total_tokens, D]
#             logits = torch.matmul(target, predicted.t()) / self.temperature
#             labels = torch.arange(logits.size(0), device=logits.device)
#             ce_loss = F.cross_entropy(logits, labels)
#             if not self.pretraining:
#                 return 0.2 * ce_loss + 0.8 * mse_loss
#             else:
#                 return ce_loss +  mse_loss

#         else:
#             # inter-modality regression
#             z    = self.encoder(x)         # [B, latent]
#             pred = self.decoder(z)         # [B, D]
#             mse_loss = F.mse_loss(pred, y)
#             # return mse_loss
#             # y = F.normalize(y, dim=-1)
#             # pred = F.normalize(pred, dim=-1)
            
#             y = y.reshape(-1, y.size(-1))        # [total_tokens, D]
#             pred = pred.reshape(-1, pred.size(-1)) # [total_tokens, D]
#             # y = torch.sign(self.hash(y))
#             # pred = torch.sign(self.hash(pred))
#             logits = torch.matmul(y, pred.t()) / self.temperature
#             # diag_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
#             # logits.masked_fill_(diag_mask, float('-inf'))
#             labels = torch.arange(logits.size(0), device=x.device).long()
#             # labels[0::2] += 1
#             # labels[1::2] -= 1
#             ce_loss = F.cross_entropy(logits, labels, reduction="mean").mean() 
            
#             if not self.pretraining:
#                 return 0.2 * ce_loss + 0.8 * mse_loss
#             else:
#                 return ce_loss +  mse_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCoTokenLoss(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        encoder_hidden: int = 256,
        proj_dim: int = 64,
        temperature: float = 0.1,
        queue_size: int = 65536,
        momentum: float = 0.999,
    ):
        super().__init__()
        self.temperature = temperature
        self.m = momentum
        self.queue_size = queue_size

        # --- query (online) encoders ---
        self.encoder = nn.Sequential(
            nn.Linear(dim, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, proj_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(proj_dim, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, dim),
        )

        # --- key (momentum) encoders: same architecture ---
        self.encoder_k = nn.Sequential(
            nn.Linear(dim, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, proj_dim),
        )
        self.decoder_k = nn.Sequential(
            nn.Linear(proj_dim, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, dim),
        )

        # initialize key networks to match query
        for q, k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            k.data.copy_(q.data)
            k.requires_grad = False
        for q, k in zip(self.decoder.parameters(), self.decoder_k.parameters()):
            k.data.copy_(q.data)
            k.requires_grad = False

        # create the queue buffer (dim, K) and pointer
        self.register_buffer("queue", torch.randn(proj_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # mask token for MAE-style masking
        self.mask_token: Optional[nn.Parameter] = None
        self.pretraining = False

    @torch.no_grad()
    def _momentum_update_key(self):
        # EMA update of key encoder
        for q, k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            k.data = k.data * self.m + q.data * (1. - self.m)
        for q, k in zip(self.decoder.parameters(), self.decoder_k.parameters()):
            k.data = k.data * self.m + q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        # gather keys across GPUs if needed…
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # replace entries [ptr: ptr + batch_size]
        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[:, ptr:end] = keys.T
        else:
            # wrap-around
            first = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :batch_size - first] = keys[first:].T
        self.queue_ptr[0] = (end % self.queue_size)

    def random_masking(self, x: torch.Tensor, mask_ratio: float = 0.3):
        # same as before…
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        # init mask token
        if self.mask_token is None or self.mask_token.shape[-1] != D:
            mt = torch.zeros(D, device=x.device)
            self.mask_token = nn.Parameter(mt)
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        mask_tokens = self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, N - len_keep, D)
        x_merged = torch.cat([x_keep, mask_tokens], dim=1)
        x_masked = torch.gather(
            x_merged, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )
        mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        mask[:, :len_keep] = False
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        # only doing masked‐MAE + MoCo‐style contrast here
        assert y is None, "MoCoTokenLoss only supports the self-supervised branch"

        # 1) mask, encode (query)
        mask_ratio = torch.rand(1).item() * 0.5 + 0.25
        x_masked, mask = self.random_masking(x, mask_ratio)
        q_proj = self.encoder(x_masked)       # [B, N, proj_dim]
        q_dec  = self.decoder(q_proj)         # [B, N, D]

        # 2) reconstruction loss
        target = x[mask]
        pred   = q_dec[mask]
        mse_loss = F.mse_loss(pred, target)

        target = F.normalize(target, dim=-1).detach()              # [M, C]
        pred = F.normalize(pred, dim=-1)


        # 5) compute logits against positives + the queue
        # positive logits: (q * k).sum(dim=-1, keepdim=True)
        # negative logits: q @ queue  → [M, K]
        l_neg = pred @ target.T                # [M, queue_size]
        logits /= self.temperature

        # 6) InfoNCE labels: positive is index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # 7) cross‐entropy
        ce_loss = F.cross_entropy(logits, labels)

        # 8) dequeue & enqueue
        self._dequeue_and_enqueue(k)

        # 9) final loss
        if not self.pretraining:
            return 0.2 * ce_loss + 0.8 * mse_loss
        else:
            return ce_loss + mse_loss




class TokenLoss(nn.Module):
    def __init__(self, 
        dim: int = 512,
        encoder_hidden: int = 256,
        proj_dim: int = 64,
        temperature: float = 0.1,
        queue_size: int = 65536,
        momentum: float = 0.999):
        super().__init__()
        self.temperature = temperature
        
        self.m = momentum
        self.queue_size = queue_size
        # encoder/decoder as before…
        D = dim
        self.encoder = nn.Sequential(
            nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, D),
        )
        # self.hash = nn.Sequential(
        #     nn.Linear(D,64),
        #     nn.Tanh()
        #     )

        # mask token for MAE-style masking
        self.mask_token: Optional[nn.Parameter] = None
        self.pretraining = False
        # we’ll lazily create this once we see an input of dim D
        self.mask_token: Optional[nn.Parameter] = None

    # def half_masking(self, x: torch.Tensor):
    #     # x: [B, D]
    #     B, D = x.shape
    #     split_idx = D // 2
    #     x_context = x.clone()
    #     x_context[:, split_idx:] = 0  # Mask the second half (set to zero)
    #     x_target = x[:, split_idx:]   # The unmasked (target) part for loss
    #     return x_context, x_target,split_idx




    def random_masking(self, x: torch.Tensor, mask_ratio: float = 0.3):
        # now expect x: [B, N, D]
        assert x.dim() == 3, "Expected [B, N, D]"
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        # 1) compute which tokens to keep
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle  = torch.argsort(noise, dim=1)
        ids_restore  = torch.argsort(ids_shuffle, dim=1)
        ids_keep     = ids_shuffle[:, :len_keep]  # [B, len_keep]
        x_keep       = torch.gather(
            x, 1,
            ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, len_keep, D]

        # 2) lazy-init mask_token at correct D
        if self.mask_token is None or self.mask_token.shape[-1] != D:
            # create as [D], on correct device
            mt = torch.zeros(D, device=x.device)
            self.mask_token = nn.Parameter(mt)
            nn.init.trunc_normal_(self.mask_token, std=0.02)

        # 3) build mask_tokens of shape [B, N_masked, D]
        mask_tokens = (
            self.mask_token
                .unsqueeze(0)      # [1, D]
                .unsqueeze(0)      # [1, 1, D]
                .expand(B, N - len_keep, D)
        )

        # 4) merge and unshuffle
        x_merged = torch.cat([x_keep, mask_tokens], dim=1)  # [B, N, D]
        x_masked = torch.gather(
            x_merged, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, N, D]

        # 5) boolean mask of masked positions
        mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        mask[:, :len_keep] = False
        mask = torch.gather(mask, 1, ids_restore)

        return x_masked, mask, ids_restore

    def half_masking(self,x: torch.Tensor):
        B, D = x.shape
        split = D // 2
        x_ctx = x.clone()
        x_ctx[:, split:] = 0        # zero out second half
        mask = torch.zeros_like(x)  # [B, D]
        mask[:, split:] = 1.0       # 1.0 on the masked dims
        return x_ctx, mask

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        # x: [B, N, D]
        if y is None:
            # mask_ratio = np.random.uniform(0.25, 0.75)
            mask_ratio = torch.rand(1).item() * 0.25 + 0.75
            # x_masked, mask, _ = self.random_masking(x,mask_ratio=mask_ratio)
            context, mask = self.half_masking(x)  # Both are [B, N//2, D]

            z = self.encoder(context)
            predicted = self.decoder(z)
            
            # predicted = predicted[:, split_idx:]
            target = x
            mse_loss = F.mse_loss(predicted, target)
          
            # logits = torch.matmul(q, k.t()) / self.temperature                # [M, queue_size]

            target = F.normalize(target, dim=-1).detach()
            predicted = F.normalize(predicted, dim=-1)
            # target = target.reshape(-1, target.size(-1))        # [total_tokens, D]
            # predicted = predicted.reshape(-1, predicted.size(-1)) # [total_tokens, D]
            # logits = torch.matmul(target, predicted.t()) / self.temperature
            # print(target.shape,predicted.shape)
            sim = predicted @ target.T
            sim /= self.temperature
            # sim = torch.matmul(predicted, target.t())
            labels = torch.arange(sim.size(0), device=sim.device)
            ce_loss = F.cross_entropy(sim, labels)
            if not self.pretraining:
                return 0.4 * ce_loss + 0.6 * mse_loss
            else:
                return ce_loss +  mse_loss

        else:
            # inter-modality regression
            z    = self.encoder(x)         # [B, latent]
            pred = self.decoder(z)         # [B, D]
            mse_loss = F.mse_loss(pred, y)
            y = F.normalize(y, dim=-1)
            pred = F.normalize(pred, dim=-1)
            y = y.reshape(-1, y.size(-1))        # [total_tokens, D]
            pred = pred.reshape(-1, pred.size(-1)) # [total_tokens, D]
            logits = torch.matmul(y, pred.t()) / self.temperature
            labels = torch.arange(logits.size(0), device=x.device).long()
            ce_loss = F.cross_entropy(logits, labels)
            if not self.pretraining:
                return 0.4 * ce_loss + 0.6 * mse_loss
            else:
                return ce_loss +  mse_loss


class Contrastive(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(
        self, h1: torch.Tensor, h2: torch.Tensor
    ) -> torch.Tensor:
        if h1.numel() == 0 or h2.numel() == 0:
            return h1.new_tensor(0.0)  # missing modality ⇒ 0 loss

        # Normalize the embeddings
        h1 = F.normalize(h1, dim=-1, eps=1e-6)
        h2 = F.normalize(h2, dim=-1, eps=1e-6)
        # Compute cosine similarity (dot product of normalized vectors)
        logits = torch.matmul(h1, h2.T) / self.temperature  # [B, B]

        # Ground truth labels: match i-th h1 with i-th h2
        labels = torch.arange(h1.size(0), device=h1.device)

        # Cross-entropy loss
        contrastive_loss = self.ce(logits, labels)

        # Optional regularization: balance the embeddings
        # bal = h1.mean(0)
        # reg = self.mse(bal, torch.zeros_like(bal)) - self.mse(h1, torch.zeros_like(h1))

        # Total loss = contrastive + regularization
        return contrastive_loss# + reg




def av_sync_contrastive_loss(
         v_emb: torch.Tensor,
         a_emb: torch.Tensor,
         temperature: float = 0.07,
         symmetric: bool = True
 ) -> torch.Tensor:
    v = F.normalize(v_emb, dim=-1, eps=1e-2)
    a = F.normalize(a_emb, dim=-1, eps=1e-2)
    logits = v @ a.T
    logits = logits / temperature
    targets = torch.arange(v.size(0), device=v.device)
    loss_va = F.cross_entropy(logits, targets)
    if symmetric:
        loss_av = F.cross_entropy(logits.T, targets)
        return 0.5 * (loss_va + loss_av)
    return loss_va

# -----------------------------------------------------------------------------
# Decoders
# -----------------------------------------------------------------------------

# ───────────────────────── helpers ────────────────────────────

def pair(t):
    """Ensure tuple."""
    return (t, t) if not isinstance(t, tuple) else t



def triple(t):
    """Ensure a **3‑tuple** (T, H, W)."""
    if isinstance(t, tuple):
        if len(t) == 3:
            return t
        if len(t) == 2:  # (H, W) provided – assume temporal = 1
            return (1, *t)
    return (t, t, t)

# ───────────────────────── building blocks ────────────────────

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.g, self.b)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim: int, mult: int = 4):
    inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner * 2, bias=False),
        GEGLU(),
        nn.Linear(inner, dim, bias=False),
    )

class TransformerDecoderBlock(nn.Module):
    """Self‑attn → (optional) cross‑attn → FFN."""

    def __init__(self, dim: int, *, heads: int = 8, dim_head: int = 64, ff_mult: int = 4, use_cross_attn: bool = False):
        super().__init__()
        self.self_attn = Attention(dim, heads=heads, dim_head=dim_head)
        self.cross_attn = Attention(dim, heads=heads, dim_head=dim_head) if use_cross_attn else None
        self.ff = FeedForward(dim, mult=ff_mult)

    def forward(self, x: torch.Tensor, *, context: Optional[torch.Tensor] = None):
        x = self.self_attn(x) + x
        if self.cross_attn is not None and context is not None:
            x = self.cross_attn(x, context=context) + x
        x = self.ff(x) + x
        return x


class TransformerPatchDecoder(nn.Module):
    """Stack of Decoder blocks shared by Audio & Video heads."""

    def __init__(self, dim: int, depth: int, *, use_cross_attn: bool):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(dim, use_cross_attn=use_cross_attn) for _ in range(depth)
        ])
        self.norm = LayerNorm(dim)

    def forward(self, x: torch.Tensor, *, context: Optional[torch.Tensor] = None):
        for blk in self.layers:
            x = blk(x, context=context)
        return self.norm(x)

 #................... Basic Decoders .............

def triple(t):
    if isinstance(t, tuple):
        if len(t) == 3:
            return t
        if len(t) == 2:
            return (1, *t)
    return (t, t, t)



class VideoDecoder(nn.Module):
    """Linear projection tokens → video patches (no Transformer)."""

    def __init__(
        self,
        dim: int,
        video_patch_size: Union[int, Tuple[int,int,int]] = (4,16,16),
        video_channels: int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = triple(video_patch_size)
        p_t, p_h, p_w = self.patch_size
        self.video_channels = video_channels
        patch_dim = p_t * p_h * p_w * video_channels
        self.to_pixels = nn.Linear(dim, patch_dim, bias=False)

    def forward(self, tokens: torch.Tensor, *, video_shape: Tuple[int,int,int]):
        """tokens [B,N,D] → reconstructed video [B,C,T,H,W]"""
        T, H, W = video_shape
        p_t, p_h, p_w = self.patch_size
        assert T % p_t == 0 and H % p_h == 0 and W % p_w == 0
        g_t, g_h, g_w = T // p_t, H // p_h, W // p_w

        B, N, _ = tokens.shape
        pixels = self.to_pixels(tokens)                    # [B,N,patch_dim]
        video  = pixels.view(B, g_t, g_h, g_w,
                             self.video_channels, p_t, p_h, p_w)
        video  = video.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return video.view(B, self.video_channels, T, H, W)


class MFCCDecoder(nn.Module):
    """Linear projection tokens → MFCC patch grid."""

    def __init__(
        self,
        dim: int,
        audio_patch_size: Union[int, Tuple[int,int]] = (40,16),
    ) -> None:
        super().__init__()
        self.patch_size = pair(audio_patch_size)
        p_c, p_t = self.patch_size   # coeffs × frames per patch
        patch_dim = p_c * p_t
        self.to_mfcc = nn.Linear(dim, patch_dim, bias=False)

    def forward(self, tokens: torch.Tensor, *, audio_shape: Tuple[int,int]):
        """tokens [B,N,D] → MFCC [B,C,T] where C=n_coeffs"""
        n_coeffs, T_frames = audio_shape
        p_c, p_t = self.patch_size
        assert n_coeffs % p_c == 0 and T_frames % p_t == 0
        g_c, g_t = n_coeffs // p_c, T_frames // p_t

        B, N, _ = tokens.shape
        coeffs = self.to_mfcc(tokens)                       # [B,N,patch_dim]
        mfcc   = coeffs.view(B, g_c, g_t, p_c, p_t)
        mfcc   = mfcc.permute(0, 3, 1, 4, 2).contiguous()
        return mfcc.view(B, n_coeffs, T_frames)

class LandmarkDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_frames: int = 72,
        num_landmarks: int = 478,
        hidden_dim: int = 256
    ):
        """
        Args:
          dim             – input token embedding size (e.g. 512)
          num_frames      – number of video frames (72)
          num_landmarks   – # of (x,y) points per frame (478)
          hidden_dim      – width of the MLP bottleneck
        """
        super().__init__()
        self.num_frames    = num_frames
        self.num_landmarks = num_landmarks

        # two‐layer MLP: dim → hidden_dim → (F*P*2)
        self.mlp = nn.Sequential(
            nn.Linear(dim,     hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_frames * num_landmarks * 2)
        )

    def forward(self, v_tok: torch.Tensor) -> torch.Tensor:
        """
        v_tok: [B, N, D]  – video patch tokens from your encoder
        returns:
          landmarks_pred: [B, F, P, 2]
        """
        B, N, D = v_tok.shape

        # 1) collapse token dimension (global pooling)
        x = v_tok.mean(dim=1)           # → [B, D]

        # 2) MLP to all landmarks
        out = self.mlp(x)               # → [B, F*P*2]

        # 3) reshape into (frames, points, xy)
        out = out.view(
            B,
            self.num_frames,
            self.num_landmarks,
            2
        )
        return out
class TransformerLandmarkDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 4,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        num_frames: int = 72,
        num_landmarks: int = 478,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_landmarks = num_landmarks
        self.grid = num_frames  # assuming 1 token per frame

        self.transformer = TransformerPatchDecoder(
            dim=dim,
            depth=depth,
            use_cross_attn=use_cross_attn,
        )
        self.to_landmarks = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_landmarks * 2)  # (x,y) for each point
        )

    def forward(
        self,
        v_tok: torch.Tensor,             # [B, N, D] from video encoder
        audio_context: Optional[torch.Tensor] = None,  # [B, M, D]
    ) -> torch.Tensor:
        """
        Returns:
          landmarks_pred: [B, F, P, 2]  where F = num_frames, P = num_landmarks
        """
        B, N, D = v_tok.shape

        # Optionally trim or pool to F tokens (1 per frame)
        if N != self.num_frames:
            # Simple fallback: mean-pool to F tokens
            v_tok = v_tok.view(B, self.num_frames, -1, D).mean(dim=2)

        x = self.transformer(v_tok, context=audio_context)  # [B, F, D]
        out = self.to_landmarks(x)  # [B, F, P*2]
        return out.view(B, self.num_frames, self.num_landmarks, 2)

# ───────────────────────── modality‑specific decoders ─────────

# class VideoDecoder(nn.Module):
#     """Unpatchifies tokens into video given runtime `video_shape`."""

#     def __init__(
#         self,
#         dim: int,
#         video_patch_size: Union[int, Tuple[int, int, int]] =  (4, 16, 16),
#         video_channels: int = 3,
#         depth: int = 1,
#         *,
#         use_cross_attn: bool = False,
#     ) -> None:
#         super().__init__()
#         self.patch_size = triple(video_patch_size)
#         # self.patch_size = (3,16,16)
#         p_t, p_h, p_w = self.patch_size
#         patch_dim = p_t * p_h * p_w * video_channels
#         self.video_channels = video_channels
#         self.to_pixels = nn.Sequential(LayerNorm(dim), nn.Linear(dim, patch_dim))
#         self.transformer = TransformerPatchDecoder(dim, depth, use_cross_attn=use_cross_attn)

#     def forward(
#         self,
#         tokens: torch.Tensor,            # [B, N, D]
#         *,
#         video_shape: Tuple[int, int, int],  # (T, H, W)
#         context: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         T, H, W = video_shape
#         p_t, p_h, p_w = self.patch_size
#         assert T % p_t == 0 and H % p_h == 0 and W % p_w == 0, "Video dims must divide patch dims"
#         g_t, g_h, g_w = T // p_t, H // p_h, W // p_w
#         # print("Tokens shape is ",tokens.shape, "Context shape is ", context.shape)
#         x = self.transformer(tokens, context=context)
#         patches = self.to_pixels(x)  # [B, N, patch_dim]
#         B = tokens.size(0)
#         video = patches.view(B, g_t, g_h, g_w, self.video_channels, p_t, p_h, p_w)
#         video = video.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
#         return video.view(B, self.video_channels, T, H, W)


# class AudioDecoder(nn.Module):
#     """Unpatchifies tokens into spectrogram given runtime `audio_shape`."""

#     def __init__(
#         self,
#         dim: int,
#         audio_patch_size: Union[int, Tuple[int, int]] = 16,
#         depth: int = 1,
#         *,
#         use_cross_attn: bool = False,
#     ) -> None:
#         super().__init__()
#         self.patch_size = pair(audio_patch_size)
#         p_f, p_t = self.patch_size
#         patch_dim = p_f * p_t
#         self.to_spec = nn.Sequential(LayerNorm(dim), nn.Linear(dim, patch_dim))
#         self.transformer = TransformerPatchDecoder(dim, depth, use_cross_attn=use_cross_attn)

#     def forward(
#         self,
#         tokens: torch.Tensor,            # [B, N, D]
#         *,
#         audio_shape: Tuple[int, int],    # (F, T)
#         context: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         F_bins, T_spec = audio_shape
#         p_f, p_t = self.patch_size
#         assert F_bins % p_f == 0 and T_spec % p_t == 0, "Spec dims must divide patch dims"
#         g_f, g_t = F_bins // p_f, T_spec // p_t

#         x = self.transformer(tokens, context=context)
#         patches = self.to_spec(x)  # [B, N, patch_dim]
#         B = tokens.size(0)
#         spec = patches.view(B, g_f, g_t, p_f, p_t).permute(0, 3, 1, 4, 2).contiguous()
#         return spec.view(B, F_bins, T_spec)

# class MFCCDecoder(nn.Module):
#     """Unpatchifies tokens into an **MFCC coefficient map** of shape (n_coeffs, T)."""

#     def __init__(
#         self,
#         dim: int,
#         audio_patch_size: Union[int, Tuple[int, int]] = 16,
#         depth: int = 1,
#         *,
#         use_cross_attn: bool = False,
#     ) -> None:
#         super().__init__()
#         self.patch_size = pair(audio_patch_size)
#         p_c, p_t = self.patch_size  # p_c: #coeffs per patch, p_t: #frames per patch
#         patch_dim = p_c * p_t
#         self.to_mfcc = nn.Sequential(LayerNorm(dim), nn.Linear(dim, patch_dim))
#         self.transformer = TransformerPatchDecoder(dim, depth, use_cross_attn=use_cross_attn)

#     def forward(
#         self,
#         tokens: torch.Tensor,            # [B, N, D]
#         *,
#         audio_shape: Tuple[int, int],    # (n_coeffs, T)
#         context: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         n_coeffs, T_frames = audio_shape
#         p_c, p_t = self.patch_size
#         assert n_coeffs % p_c == 0 and T_frames % p_t == 0, "MFCC dims must divide patch dims"
#         g_c, g_t = n_coeffs // p_c, T_frames // p_t

#         x = self.transformer(tokens, context=context)
#         patches = self.to_mfcc(x)  # [B, N, patch_dim]
#         B = tokens.size(0)
#         mfcc = patches.view(B, g_c, g_t, p_c, p_t).permute(0, 3, 1, 4, 2).contiguous()
#         return mfcc.view(B, n_coeffs, T_frames)

# ───────────────────── simple wrappers ─────────────────────

class Audio2Video(nn.Module):
    def __init__(self, encoder: nn.Module, video_decoder: VideoDecoder):
        super().__init__()
        self.encoder = encoder
        self.vdec = video_decoder
    def forward(self, audio, dummy_video):
        _, a_tok, _, v_tok = self.encoder(audio=audio, video=dummy_video)
        # assume dummy_video.shape[2:] is (T,H,W) after permute in caller
        T, H, W = dummy_video.shape[2:]
        return self.vdec(a_tok, video_shape=(T, H, W), context=v_tok)


        
class Video2Audio(nn.Module):
    def __init__(self, encoder: nn.Module, audio_decoder: AudioDecoder):
        super().__init__()
        self.encoder = encoder
        self.adec = audio_decoder
    def forward(self, video, dummy_audio):
        _, a_tok, _, v_tok = self.encoder(audio=dummy_audio, video=video)
        F_bins, T_spec = dummy_audio.shape[-2:]
        return self.adec(v_tok, audio_shape=(F_bins, T_spec), context=a_tok)

class ImageDecoder(nn.Module):
    """2‑D patch tokens → RGB image (or arbitrary channels)."""

    def __init__(
        self,
        dim: int,
        image_shape: Tuple[int, int],  # (H, W)
        image_patch_size: Union[int, Tuple[int, int]] = 16,
        image_channels: int = 3,
        depth: int = 4,
        use_cross_attn: bool = False,
    ):
        super().__init__()
        self.patch_size = pair(image_patch_size)
        p_h, p_w = self.patch_size
        H, W = image_shape
        assert H % p_h == 0 and W % p_w == 0, "Image dims must divide patch dims"
        self.grid = (H // p_h, W // p_w)
        patch_dim = p_h * p_w * image_channels
        self.transformer = TransformerPatchDecoder(dim, depth, use_cross_attn=use_cross_attn)
        self.to_pixels = nn.Sequential(LayerNorm(dim), nn.Linear(dim, patch_dim))
        self.channels = image_channels

    def forward(self, tokens: torch.Tensor, *, context: Optional[torch.Tensor] = None):
        B = tokens.size(0)
        x = self.transformer(tokens, context=context)
        patches = self.to_pixels(x)  # [B, N, patch_dim]
        p_h, p_w = self.patch_size
        g_h, g_w = self.grid
        img = patches.view(B, g_h, g_w, self.channels, p_h, p_w).permute(0, 3, 1, 4, 2, 5).contiguous()
        return img.view(B, self.channels, g_h * p_h, g_w * p_w)

class TextDecoder(nn.Module):
    """
    Transformer‑based decoder that converts per‑token embeddings into logits
    over a target vocabulary.  It can optionally cross‑attend to another
    modality’s tokens (e.g. video → text) and share its output projection
    weights with a pre‑trained embedding layer.

    Args
    ----
    dim : int
        Embedding dimension of the input tokens.
    vocab_size : int
        Size of the vocabulary for which logits are produced.
    depth : int, default 4
        Number of Transformer decoder blocks.
    use_cross_attn : bool, default False
        If ``True``, each block includes a cross‑attention sub‑layer that can
        attend to a ``context`` passed at call time.
    tied_embedding : nn.Embedding | None, default None
        If provided, the decoder’s output projection weights are tied to this
        embedding matrix (weight tying à la GPT/BERT).

    Forward
    -------
    tokens : Tensor[B, N, dim]
        Sequence of token embeddings to decode.
    context : Tensor[B, M, dim] | None
        Optional conditioning sequence for cross‑attention.

    Returns
    -------
    logits : Tensor[B, N, vocab_size]
        Per‑token vocabulary logits.
    """
    def __init__(
        self,
        dim: int,
        vocab_size: int,
        depth: int = 4,
        *,
        use_cross_attn: bool = False,
        tied_embedding: Optional[nn.Embedding] = None,
    ) -> None:
        super().__init__()

        self.transformer = TransformerPatchDecoder(
            dim,
            depth,
            use_cross_attn=use_cross_attn,
        )

        # Output projection — optionally weight‑tied to a shared embedding
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)
        if tied_embedding is not None:
            self.to_logits.weight = tied_embedding.weight  # type: ignore[assignment]

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.transformer(tokens, context=context)
        return self.to_logits(x)  # shape: [B, N, vocab_size]


def greedy_generate(video, max_len=32):
    with torch.no_grad():
        # encode once
        _, _, _, v_tok, _ = encoder(video=video, audio=dummy_audio, text=None)

        # seed with BOS
        ids = torch.tensor([[BOS]], device=video.device)
        for _ in range(max_len):
            t_emb = encoder._encode_text(ids)               # reuse same embedding fn
            logits = text_decoder(t_emb, context=v_tok)[:,-1]  # last step
            next_id = logits.argmax(-1, keepdim=True)        # greedy
            ids = torch.cat([ids, next_id], dim=-1)
            if next_id.item() == EOS: break
    return tokenizer.decode(ids.squeeze().tolist()[1:-1])    # skip BOS/EOS

    #     loss = F.cross_entropy(
    #         logits.reshape(-1, logits.size(-1)),
    #         decoder_target_ids.reshape(-1),
    #         ignore_index=PAD
    # )
# -----------------------------------------------------------------------------
# Moderator model
# -----------------------------------------------------------------------------

class MMModerator(nn.Module):
    def __init__(self, *, dim: int = 512, device: torch.device | str = "cpu",pretraining=True):
        super().__init__()
        self.device = torch.device(device)
        self.pretraining = pretraining
        self.encoder = MaskEncoder(dim=dim, depth=6, heads=6, num_fusion_tokens=32)
        # self.encoder = MaskEncoder(dim=dim, depth=8, heads=6, num_fusion_tokens=32)
        # self.video_decoder = LandmarkDecoder(dim, video_patch_size=(4, 16, 16))
        # self.video_decoder = LandmarkDecoder(
        #     dim=dim,
        #     num_frames=72,
        #     num_landmarks=478,
        #     hidden_dim=256
        # )
        # self.video_decoder = TransformerLandmarkDecoder(
        #     dim=dim,
        #     num_frames=72,
        #     num_landmarks=478,
        #     use_cross_attn=True
        # )

        # self.audio_decoder = MFCCDecoder(dim, audio_patch_size=(40, 16))
        # single image‑to‑tokens conv
        p = 16
        # self.image_to_tokens = nn.Sequential(
        #     nn.Conv2d(3, dim, kernel_size=p, stride=p),
        #     Rearrange("b d h w -> b (h w) d"),
        # )
        # self.landmark_encoder = nn.Sequential(
        #     nn.Linear(478*2,512)
        #     )
        self.fc_norm = nn.LayerNorm(dim)
        self.fusion_attn = nn.MultiheadAttention(dim, 4, batch_first=True, device=self.device)
        self.proj_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            # nn.BatchNorm1d(dim // 2),       
            nn.LayerNorm(dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),  
            nn.Linear(dim // 2, dim // 2),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(dim // 2, dim // 2),
            # nn.BatchNorm1d(dim // 2),   
            nn.LayerNorm(dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),              
            nn.Linear(dim // 2, 1)     
            # apply sigmoid externally or use BCEWithLogitsLoss
        )

        # self.logit_scale = nn.Parameter(torch.ones(1))   # inside MMModerator
        # self.lpips_fn = lpips.LPIPS(net='alex').to(device)   # or net='vgg'
        self.predictor = nn.Linear(512,512)
        self.contrastive = Contrastive()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.token_simclr = TokenLoss()
        # self.intra_simclr = PatchContrastiveLoss()
      

    # --------------------------------------------------------------------- util
    def _maybe_tokens(self, images: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if not present(images) else self.image_to_tokens(images)
  # ---- safe pooling (no tokens → zeros) ---- #
    def _safe_mean(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            # zero vector in normalised space
            return torch.zeros(x.size(0), x.size(2), device=x.device)

        return x.mean(dim=1)
    # ---------------------------------------------------------------- forward
    def forward(
        self,
        mfcc: Optional[torch.Tensor],
        mfcc_aug: Optional[torch.Tensor],
        audio: Optional[torch.Tensor],
        video: torch.Tensor,  # video is mandatory here
        video_aug: Optional[torch.Tensor] = None,
        landmarks: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and total loss."""
        img_tok = None
        if video_aug is not None: 
            video_aug = video_aug.permute(0,2,1,3,4) 
        if video is not None: 
            video = video.permute(0,2,1,3,4)
        if flow is not None:
            flow = flow.permute(0,2,1,3,4)
        # if images is not None:
        #     img_tok = self._maybe_tokens(images)

        # encode
        pooled, a_tok, f_tok, v_tok = self.encoder(
            audio=mfcc,
            video=video
        )
        
        _, a_tok_b, _, v_tok_b = self.encoder(
            audio=mfcc_aug,
            video=video_aug
        )
        # fused_tok, a_token, v_token, o_token, t_token, i_token = pooled.split(1, dim=1)  # use first token as CLS
        # audio_tokens = self._safe_mean(a_tok)
        # video_tokens = self._safe_mean(v_tok)
        # fused_tokens = self._safe_mean(f_tok)
        

        # simple fusion: avg of encoder fusion token + attention pooling over union
        # combined = torch.cat(
        #     [v_tok,f_tok,a_tok], dim=1
        # )
        # print(v_tok.shape, a_tok.shape)
        # v_tok = v_tok.contiguous()
        # a_tok = a_tok.contiguous()
        # f_tok = f_tok.contiguous()

       
        if not self.pretraining:
            va_out, _ = self.fusion_attn(v_tok, a_tok, a_tok)
            fva_out, _ = self.fusion_attn(va_out, f_tok, f_tok)
            # print(av_embds.shape)
            # fva_out = fva_out.transpose(0, 1)

            assert torch.isfinite(fva_out).all(), "fva_out has NaNs"
            avf_out = self._safe_mean(fva_out)
            assert torch.isfinite(fva_out).all(), "fva_out has NaNs"
            logits_cls = self.proj_head(avf_out)
            logits_cls = self.cls_head(logits_cls)
        else:
            logits_cls = None
        # print(avf_out.shape)
       
        # logits_cls = self.proj_head(avf_out)
        # logits_cls = self.cls_head(logits_cls)

        # ---------------- contrastive losses ----------------
        

        tokens: Dict[str, torch.Tensor] = {
        # "v": self._safe_mean(v_tok),
        # "a": self._safe_mean(a_tok),
        "v": v_tok,
        "a": a_tok,
        # "t": t_tok,
        }
        self.token_simclr.pretraining = self.pretraining
        # prepare empty losses dict
        losses: Dict[str, Dict[str, torch.Tensor]] = {
            "intra": {},  # per-modality SimCLR
            "inter": {},  # cross-modal contrastive
            "lipsyncLoss":{},
            "reconstruction":{},
            "cls_loss":{}
        }
        # if not self.pretraining:
        # 1) intra‐modal SimCLR losses
        for name, tok in tokens.items():
            if tok.numel() > 0:
                tok = self._safe_mean(tok)
                losses["intra"][name] = self.token_simclr(tok)

        # 2) collapse tokens to a [B, D] mean embedding
        mean_toks = {
            name: tok
            for name, tok in tokens.items()
        }

        # 3) inter‐modal contrastive pairs
        inter_pairs: List[Tuple[str, str]] = [
            ("v", "a"),
            ("a", "v"),
            # ("v", "t"),
            # ("a", "t"),
        ]
        for m1, m2 in inter_pairs:
            
            m1 = mean_toks[m1]
            m2 = mean_toks[m2]
            e1, e2 = m1, m2
            if e1.numel() > 0 and e2.numel() > 0:
                # both directions
                e1 = self._safe_mean(e1)
                e2 = self._safe_mean(e2)
                # losses["inter"][f"{m1}_{m2}"] = self.contrastive(e1, e2)
                losses["inter"][f"{m1}_{m2}"] = self.token_simclr(e1, e2)
                
                # losses["inter"][f"{m2}_{m1}"] = self.contrastive(e2, e1)


        # a_frame = self._safe_mean(a_tok)
        # v_frame = self._safe_mean(v_tok)
        # losses["lipsyncLoss"]["loss"] = av_sync_contrastive_loss(v_frame,a_frame)
        
        ##################################
        #       reconstruction loses    ##
        ##################################
        T, H, W = video.shape[2:]
        n_coeffs, T_frames = mfcc.shape[-2:]
        # synthetic_video = self.video_decoder(v_tok, video_shape=(T, H, W), context=a_tok)
        # synthetic_audio = self.audio_decoder(a_tok, audio_shape=(n_coeffs, T_frames), context=v_tok)
        # landmarks = self.landmark_encoder(landmarks.flatten(start_dim=2,end_dim=3))
        # landmarks,_ = self.fusion_attn(landmarks, v_tok, v_tok)
        # landmarks = self._safe_mean(landmarks)
        # max_len = landmarks.shape[-1]
        # print(landmarks.shape)
        # exit()

        # synthetic_video = self.video_decoder(v_tok, audio_context=a_tok)
        
        # synthetic_audio = self.audio_decoder(a_tok, audio_shape=(n_coeffs, T_frames))
        
        # e1 = self._safe_mean(v_tok)
        # e2 = self._safe_mean(v_tok_b)

        
        # a_tok = F.normalize(a_tok, dim=-1)
        # a_tok_b = F.normalize(a_tok_b, dim=-1)
        # cos_loss_a = 1 - F.cosine_similarity(a_tok, a_tok_b, dim=-1).mean()
        
        # e1 = self._safe_mean(a_tok)
        # e2 = self._safe_mean(a_tok_b)
        # loss_mse_a  = F.mse_loss(synthetic_audio,  mfcc)
        # loss_mse_v = loss_mse_v.mean()
        # loss_mse_a = loss_mse_a.mean()


        # losses["reconstruction"]["audio"] = self.token_simclr(a_tok_b, a_tok)
        # losses["reconstruction"]["video"] = self.token_simclr(v_tok_b, v_tok)  # KL divergence loss


        T = 2.0  # Temperature parameter
        alpha = 0.8  # Weighting factor between losses
        # Assuming 'student_logits' and 'teacher_logits' are the output logits from your models
        if audio !=None:
            augmented = self.predictor(self._safe_mean(a_tok_b)).detach()
            ori = self.predictor(self._safe_mean(a_tok))
            teacher_log_probs = F.log_softmax(augmented / T, dim=1)
            student_probs = F.softmax(ori / T, dim=1)
            kl_loss_audio = F.kl_div(teacher_log_probs, student_probs, reduction='batchmean') * (T * T)
            # Combine losses
            # total_loss = alpha * kl_loss + (1 - alpha) * (losses["reconstruction"]["audio"])

            losses["reconstruction"]["kl_divergence_audio"] =  alpha * kl_loss_audio + (1 - alpha) * (self.token_simclr(a_tok_b, a_tok))# KL divergence loss
        if video!=None:
            augmented = self.predictor(self._safe_mean(v_tok_b)).detach()
            ori = self.predictor(self._safe_mean(v_tok))
            teacher_log_probs = F.log_softmax(augmented / T, dim=1)
            student_probs = F.softmax(ori / T, dim=1)
            kl_loss_video = F.kl_div(teacher_log_probs, student_probs, reduction='batchmean') * (T * T)
            # Combine losses
            # total_loss = alpha * kl_loss + (1 - alpha) * (losses["reconstruction"]["video"])
            
            losses["reconstruction"]["kl_divergence_video"] =  alpha * kl_loss_video + (1 - alpha) * (self.token_simclr(v_tok_b, v_tok))# KL divergence loss


        # augmented = self.predictor(self._safe_mean(a_tok_b)).detach()
        # ori = self.predictor(self._safe_mean(v_tok))
        # teacher_log_probs = F.log_softmax(augmented / T, dim=1)
        # student_probs = F.softmax(ori / T, dim=1)
        # kl_loss_audio_video = F.kl_div(teacher_log_probs, student_probs, reduction='batchmean') * (T * T)

        # losses["reconstruction"]["kl_divergence_audio_video"] =  alpha * kl_loss_audio_video + (1 - alpha) * (self.token_simclr(self._safe_mean(a_tok_b), self._safe_mean(v_tok)))# KL divergence loss



        # augmented = self.predictor(self._safe_mean(v_tok_b)).detach()
        # ori = self.predictor(self._safe_mean(a_tok))
        # teacher_log_probs = F.log_softmax(augmented / T, dim=1)
        # student_probs = F.softmax(ori / T, dim=1)
        # kl_loss_video_audio = F.kl_div(teacher_log_probs, student_probs, reduction='batchmean') * (T * T)

        # losses["reconstruction"]["kl_divergence_video_audio"] =  alpha * kl_loss_video_audio + (1 - alpha) * (self.token_simclr(self._safe_mean(v_tok_b), self._safe_mean(a_tok)))# KL divergence loss

        
        # losses["reconstruction"]["landmarks"] = self.token_simclr(fva_out, landmarks)  
        # print(logits_cls.shape, labels.shape)
        if not self.pretraining:
            losses["cls_loss"]["loss"] = self.cls_loss(logits_cls.view(-1), labels)
            ##################################
            #       reconstruction loses    ##
            ##################################
       
        return logits_cls, losses