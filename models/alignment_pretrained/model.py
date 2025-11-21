from __future__ import annotations


import math
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision.models.video import r3d_18, R3D_18_Weights
from .masked_encoder import MaskEncoder as MaskEncoder, TokenTypes, Attention
from .decoders import VisualDecoder, AudioDecoder, VisualDecoder16x16
import lpips
from transformers import VideoMAEForVideoClassification

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

def kd_loss(output, label, teacher_output, alpha, temperature):
    """
    from:
        https://github.com/peterliht/knowledge-distillation-pytorch
        
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = alpha
    T = temperature
    
    """
    kd_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(output/T, dim=1),
                                             F.softmax(teacher_output/T, dim=1)).type(torch.FloatTensor).cuda(gpu)
    
    kd_loss = kd_filter * torch.sum(kd_loss, dim=1) # kd filter is filled with 0 and 1.
    kd_loss = torch.sum(kd_loss) / torch.sum(kd_filter) * (alpha * T * T)
    """
    
    kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output/T, dim=1),
                                                  F.softmax(teacher_output/T, dim=1)) * (alpha * T * T)
    
    cr_loss = nn.BCEWithLogitsLoss()(output.view(-1), label) * (1. - alpha)
    
    return kd_loss + cr_loss



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


class TokenLoss(nn.Module):
    def __init__(self, 
        dim: int = 512,
        encoder_hidden: int = 256,
        proj_dim: int = 64,
        temperature: float = 0.0,
        queue_size: int = 65536,
        momentum: float = 0.999):
        super().__init__()
        self.temperature = temperature
        self.m = momentum
        self.queue_size = queue_size
        # self.W = nn.Parameter(torch.randn(dim,dim))
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

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, detach=False) -> torch.Tensor:
        # x: [B, N, D]
        if y is None:
        
            context, mask = self.half_masking(x)  # Both are [B, N//2, D]

            z = self.encoder(context)
            predicted = self.decoder(z)
            
            # predicted = predicted[:, split_idx:]
            target = x
            mse_loss = F.mse_loss(predicted, target)
        
            # logits = torch.matmul(q, k.t()) / self.temperature                # [M, queue_size]

            target = F.normalize(target, dim=-1).detach()
            predicted = F.normalize(predicted, dim=-1)

            sim = predicted @ target.T
            # new unknown
            # sim  = torch.mean(predicted * target, dim=-1)  
            # sim /= self.temperature
            # sim = torch.matmul(predicted, target.t())
            labels = torch.arange(sim.size(0), device=sim.device)
            # print(sim.shape)
            ce_loss = F.cross_entropy(sim, labels)
            if not self.pretraining:
                return 0.4 * ce_loss + 0.6 * mse_loss
            else:
                return ce_loss +  mse_loss

        else:
            gen = x.detach()
            
            z    = self.encoder(y)         # [B, latent]
            pred = self.decoder(z)         # [B, D]
            mse_loss = F.mse_loss(pred, gen)
            x = F.normalize(x, dim=-1)
            pred = F.normalize(pred, dim=-1)
            if len(pred.shape) > 2:
                x = x.reshape(-1, x.size(-1))        # [total_tokens, D]
                pred = pred.reshape(-1, pred.size(-1)) # [total_tokens, D]
            # # logits = torch.matmul(y, pred.t()) #/ self.temperature

            # print(pred.shape, x.shape)
            # logits  = torch.sum(pred * x, dim=-1) # this worked:
            # print(pred.shape, x.T.shape)
            logits = pred @ x.T

            labels = torch.arange(logits.size(0), device=x.device, dtype=torch.long)
            
            # print(logits.shape, labels.shape)
            # print(logits, labels)
            
    
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
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.6, use_logits=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_logits = use_logits

    def forward(self, logits, targets):
        # BCE per sample
        if self.use_logits:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        else:
            bce = F.binary_cross_entropy(logits, targets, reduction='none')

        # p_t = exp(−BCE)  gives probability of the true class
        pt = torch.exp(-bce)

        # α_t: alpha for positives, (1−alpha) for negatives
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # focal term
        focal_term = (1 - pt) ** self.gamma

        loss = alpha_t * focal_term * bce
        return loss.mean()


class BalancedBinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, use_logits=True):
        super().__init__()
        self.gamma = gamma
        self.use_logits = use_logits

    def forward(self, logits, targets):
        # BCE
        if self.use_logits:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        else:
            bce = F.binary_cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce)

        # compute batch frequencies
        N_pos = targets.sum()
        N_neg = (1 - targets).sum()
        # avoid div-by-zero
        alpha_pos = N_neg / (N_pos + N_neg + 1e-6)
        alpha_neg = N_pos / (N_pos + N_neg + 1e-6)

        # class-dependent alpha
        alpha_t = targets * alpha_pos + (1 - targets) * alpha_neg
        focal_term = (1 - pt) ** self.gamma
        loss = alpha_t * focal_term * bce
        return loss.mean()




class MMModerator(nn.Module):
    def __init__(self, *, dim: int = 512, device: torch.device | str = "cpu",pretraining=False, num_classes=11):
        super().__init__()
        self.device = torch.device(device)
        self.pretraining = pretraining
        self.encoder = MaskEncoder(dim=dim, depth=6, heads=4, num_fusion_tokens=16,video_temporal_patch_size=4,device=device) # Gpu 4 25.42 million parameters
        self.video_decoder = VisualDecoder16x16(n_frames=16, tubelet_size=4, embed_dim=512, depth=6, num_heads=4, encoder_embed_dim=512)
        

        self.fc_norm = nn.LayerNorm(dim)
        self.fusion_attn = nn.MultiheadAttention(dim, 4, batch_first=True, device=self.device)
        self.proj_head = nn.Sequential(
            nn.Linear(dim, dim),
            # nn.BatchNorm1d(dim ),  
            nn.Dropout(0.2),  
            nn.LayerNorm(dim ),
            nn.LeakyReLU(),     
            nn.Linear(dim  , dim)
        )
        self.classes_stream = nn.Sequential(
            nn.Linear(dim, dim),
            # nn.BatchNorm1d(dim ),  
            nn.Dropout(0.2),  
            nn.LayerNorm(dim ),
            nn.LeakyReLU(),     
            nn.Linear(dim, num_classes)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(dim , dim),  
            nn.BatchNorm1d(dim ), 
            nn.LeakyReLU(),
            nn.Dropout(0.2),    
            # nn.LayerNorm(dim ),
            nn.Linear(dim , dim ),
            # nn.LeakyReLU(),
            nn.Dropout(0.2),    
            #nn.Sigmoid(),
            # nn.ReLU(),          
            nn.Linear(dim , 1)     
        )
        # self.logit_scale = nn.Parameter(torch.ones(1))   # inside MMModerator
        # self.lpips_fn = lpips.LPIPS(net='alex').to(device)   # or net='vgg'
        self.predictor = nn.Linear(512,512)
        self.predictor_intra = nn.Linear(512,512)
        self.predictor_inter = nn.Linear(512,512)
        self.predictor_recon = nn.Linear(512,512)
        self.contrastive = Contrastive()
        # self.cls_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = BinaryFocalLoss()
        self.multi_cls_loss = nn.CrossEntropyLoss()
        self.token_simclr = TokenLoss()
        
        self.norm_pix_loss = True
        
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
    
    @torch.no_grad()
    def _momentum_update_teacher(self):
        """
        EMA update: teacher_params ← m*teacher_params + (1-m)*student_params
        """
        for student_p, teacher_p in zip(self.encoder.parameters(),
                                        self.teacher_encoder.parameters()):
            teacher_p.data = teacher_p.data * self.momentum + student_p.data * (1.0 - self.momentum)

    # ---------------------------------------------------------------- forward
    
    def forward_mse_loss(self, target, pred):
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(var + 1e-6)
        
        loss = (pred - target).pow(2)
        loss = loss.mean()
    
        return loss

    def forward(
        self,
        mfcc: Optional[torch.Tensor] = None,
        mfcc_aug: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,  # video is mandatory here
        video_aug: Optional[torch.Tensor] = None,
        landmarks: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        images_aug: Optional[torch.Tensor] = None,
        # fft_magnitude:Optional[torch.Tensor] = None,
        multi_label: Optional[torch.Tensor] = None,
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
        #     img_tok = torch.zeros_like()
        text_aug = text
        # print(mfcc.shape)
        # encode
        # print(images.shape)
        
        pooled, af_tok,a_tok, vf_tok, v_tok,tf_tok, t_tok, if_tok, i_tok = self.encoder(
            audio=mfcc,
            video=video_aug, 
            text_tokens=landmarks,
            image=images
        )
        

        af_tok,a_tok, vf_tok, v_tok_,tf_tok, t_tok, if_tok, i_tok = pooled.unbind(dim=1)
    
        mse_loss_video = 0

        # # if self.training:
        v_tok = self.predictor(v_tok)
        video_recon = self.video_decoder(v_tok)
        video_recon = self.video_decoder.unpatch_to_img(video_recon)
        # # print(video_recon.shape, multi_label.shape)
        # if self.training:
        #     mse_loss_video = self.forward_mse_loss(multi_label.permute(0,2,1,3,4), video_recon)
        


        # ---------------- contrastive losses ----------------
        losses: Dict[str, Dict[str, torch.Tensor]] = {
            "intra": {},  # per-modality SimCLR
            "inter": {},  # cross-modal contrastive
            "lipsyncLoss":{},
            "reconstruction":{},
            "moe_loss":{},
            "cls_loss":{}
        }

        if not self.pretraining:
            va_out, _ = self.fusion_attn(pooled, pooled, pooled)
            assert torch.isfinite(va_out).all(), "f_out has NaNs"
            classes_logits = self.fc_norm(va_out)
            # classes_logits = self.classes_stream(classes_logits)
            projection_out = self.proj_head(classes_logits)
            f_out = self._safe_mean(projection_out)
            logits_cls = self.cls_head(f_out) 
            classes_logits = self._safe_mean(classes_logits)
            
            if self.training:
                labels = torch.concat([labels,torch.zero_like(labels)],dim=0)
            losses["cls_loss"]["loss"] = self.cls_loss(logits_cls.view(-1), labels)
           
        else: 
            logits_cls = torch.randn(video.shape[0], device=video.device)

        losses["reconstruction"]["loss"] = mse_loss_video

        return logits_cls, losses, labels, video_recon

