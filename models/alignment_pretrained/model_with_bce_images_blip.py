from __future__ import annotations

from torch.nn.utils.rnn import pad_sequence

from PIL import Image
import cv2
import matplotlib.pyplot as plt
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
from vit.pytorch_pretrained_vit.model import ViT
from .decoders import VisualDecoder, AudioDecoder, VisualDecoder16x16
from .image_decoder import MAEDecoder
from .unet import UNetImageDecoder
import lpips
from transformers import VideoMAEForVideoClassification
import random
from ..xbm.models.blip import create_vit, init_tokenizer, load_checkpoint
import torchvision
from ..xbm.models.med import BertConfig, BertLMHeadModel, BertModel
# from transformers import AutoTokenizer, BertLMHeadModel as BertLMHeadModelFrozen
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig
from models.LoRA import LoRA
from models.LoRABert import LoRAForBERT

from transformers import AutoModel, AutoTokenizer
from transformers import Blip2Processor, Blip2ForConditionalGeneration


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

        
def _sum_over(pred, dims):
    # stable sum across given dims
    for d in sorted(dims, reverse=True):
        pred = pred.sum(dim=d)
    return pred

def dice_loss(pred, tgt, eps=1e-6):
    """
    Works for 4D [B,C,H,W] and 5D [B,C,T,H,W].
    Reduces over all non-batch dims.
    """
    assert pred.shape == tgt.shape, f"shape mismatch: {pred.shape} vs {tgt.shape}"
    reduce_dims = tuple(range(1, pred.ndim))
    num = 2.0 * _sum_over(pred * tgt, reduce_dims)
    den = _sum_over(pred.pow(2), reduce_dims) + _sum_over(tgt.pow(2), reduce_dims) + eps
    return 1.0 - (num / den).mean()

def total_variation(x, include_temporal=False):
    """
    TV over spatial dims; optionally temporal for 5D.
    - 4D: [B,C,H,W]
    - 5D: [B,C,T,H,W]
    """
    if x.ndim == 4:  # [B,C,H,W]
        tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w

    elif x.ndim == 5:  # [B,C,T,H,W]
        tv_h = (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs().mean()
        tv_w = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs().mean()
        if include_temporal:
            tv_t = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).abs().mean()
            return tv_h + tv_w + tv_t
        return tv_h + tv_w

    else:
        raise ValueError(f"Unsupported ndim={x.ndim}")

def broadcast_is_real(labels: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """
    labels: [B] (0=real, 1=fake)
    like:   target tensor we want to broadcast to (4D [B,C,H,W] or 5D [B,C,T,H,W])
    returns: [B,1,1,1] or [B,1,1,1,1]
    """
    B = labels.shape[0]
    # [B] -> [B, 1, 1, 1, ...]
    
    # mask = (labels == 0).float().view(B, *([1] * (like.ndim - 1)))
    return (labels == 0).float().reshape(B, *([1] * (like.ndim - 1)))
 

def ensure_match(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Make a binary mask match ref’s rank and channels.
    - If mask is [B,H,W], turn into [B,1,H,W]
    - If ref is 5D, add temporal dim for the mask
    - If ref has C>1 and mask has C=1, repeat along channels
    """
    t = mask
    # If mask is [B,H,W] and ref is [B,C,H,W] or [B,C,T,H,W]
    if ref.ndim >= 4 and t.ndim == 3:
        t = t.unsqueeze(1)  # -> [B,1,H,W]
    # If ref is 5D and mask is 4D, add T dim
    if ref.ndim == 5 and t.ndim == 4:
        t = t.unsqueeze(2)  # -> [B,1,1,H,W] if it had [B,1,H,W]
    # Channel repeat if needed
    if t.shape[1] == 1 and ref.shape[1] > 1:
        t = t.repeat(1, ref.shape[1], *([1] * (t.ndim - 2)))
    return t
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


# class WaveAct(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self._w1 = nn.Parameter(torch.zeros(1))
#         self._w2 = nn.Parameter(torch.zeros(1))
#     def forward(self, x):
#         # keep weights around ~[0, 2] smoothly
#         w1 = F.softplus(self._w1) + 1e-4
#         w2 = F.softplus(self._w2) + 1e-4
#         x_safe = x.clamp(-100, 100)  # avoids fp16 trig corner cases
#         return w1 * torch.sin(x_safe) + w2 * torch.cos(x_safe)

class WaveAct(nn.Module):
    """
    y = x*(residual) + a*sin(ωx + φ) + b*cos(ωx + φ)

    Notes:
    - Trig evaluated in fp32 for AMP stability, cast back to input dtype.
    - softplus() keeps a,b,ω >= 0 (smoothly), preventing sign-flip explosions early on.
    - Initialize near identity by using residual=True and small a,b.
    """
    def __init__(
        self,
        residual: bool = True,
        learn_freq: bool = False,
        init_amp: float = 0.1,    # small wave at start
        clamp: float = 100.0      # avoid extreme trig args in fp16
    ):
        super().__init__()
        # log-params → softplus for a,b,ω
        self.log_a = nn.Parameter(torch.log(torch.tensor(init_amp)))
        self.log_b = nn.Parameter(torch.log(torch.tensor(init_amp)))
        self.learn_freq = learn_freq
        if learn_freq:
            self.log_w = nn.Parameter(torch.zeros(1))  # ω ≈ 0+ → after softplus add 1
        else:
            self.register_buffer("log_w", torch.zeros(1), persistent=False)
        self.phi = nn.Parameter(torch.zeros(1))        # phase
        self.residual = residual
        self.clamp = clamp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.softplus(self.log_a) + 1e-6
        b = F.softplus(self.log_b) + 1e-6
        w = (F.softplus(self.log_w) + 1.0) if self.learn_freq else 1.0

        # do trig in fp32 to avoid fp16 corner cases
        x32 = x.float().clamp(-self.clamp, self.clamp)
        t = w * x32 + self.phi
        y = a * torch.sin(t) + b * torch.cos(t)

        if self.residual:
            y = x32 + y

        return y.to(x.dtype)

class LoRALinear(nn.Module):
    """
    Standard LoRA adaptation for Linear layers:
    y = xW + α * x(AB)
    """
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, freeze: bool = True):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # frozen pretrained weight
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=not freeze)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=not freeze)
        else:
            self.bias = None

        # LoRA adapters (low-rank trainable matrices)
        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = (x @ self.A.t() @ self.B.t()) * self.scaling
        return base + lora


def apply_lora_to_self_attn(model, r: int = 8, alpha: int = 16, freeze: bool = True):
    for name, module in model.named_children():
        if "self_attn" in name:
            for sub_name, sub_module in list(module.named_modules()):
                if isinstance(sub_module, nn.Linear):
                    parent = module
                    parts = sub_name.split(".")
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], LoRALinear(sub_module, r=r, alpha=alpha, freeze=freeze))
        else:
            apply_lora_to_self_attn(module, r=r, alpha=alpha, freeze=freeze)
    return model

def _fmt_regions(regs, limit=3):
    names = [r[0] for r in regs[:limit]]
    if not names: return "across the face"
    if len(names) == 1: return names[0]
    if len(names) == 2: return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"

def _severity_phrase(sev: str, label_is_fake: bool) -> str:
    # map severity + label to phrasing
    if not label_is_fake:
        return {
            "low": "no meaningful manipulation",
            "moderate": "minor anomalies unlikely to be manipulations",
            "high": "noticeable inconsistencies but still consistent with a real image",
            "very high": "strong but likely benign inconsistencies",
        }[sev]
    return {
        "low": "subtle manipulation",
        "moderate": "clear signs of manipulation",
        "high": "pronounced deepfake artifacts",
        "very high": "extensive synthetic alterations",
    }[sev]

_SYNS = {
    "artifacts": ["artifacts", "irregularities", "anomalies", "inconsistencies"],
    "texture":   ["texture", "surface detail", "micro-detail"],
    "lighting":  ["lighting", "illumination", "shading"],
    "boundaries":["boundaries", "edges", "contours"],
    "detected":  ["detected", "observed", "identified", "evident"],
}

def _pick(k: str):  # synonym
    lst = _SYNS[k]
    return lst[random.randrange(len(lst))]


class SemanticConsistencyLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, student_emb, teacher_emb):
        sim = self.cos(student_emb, teacher_emb.detach())
        loss = 1 - sim.mean()
        return loss


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

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, detach=False) -> torch.Tensor:
        # x: [B, N, D]
        if y is None:
            y = x
        x = F.normalize(x, dim=-1)
        pred = F.normalize(y, dim=-1)
        logits = pred @ x.T
        labels = torch.arange(logits.size(0), device=x.device, dtype=torch.long)
        ce_loss = F.cross_entropy(logits, labels)
        return ce_loss


class MMModerator(nn.Module):
    def __init__(self, *, dim: int = 512, device: torch.device | str = "cpu",pretraining=False, vision_encoder=None, unet_decoder=None, num_classes=11):
        super().__init__()
        self.device = torch.device(device)
        self.pretraining = pretraining
        dim=1024
        self.encoder_bottleneck = MaskEncoder(dim=dim, depth=12, heads=6, num_fusion_tokens=16,video_temporal_patch_size=4,video_channels=3,device=device) # Gpu 4 25.42 million parameters
        # self.encoder = ViT('B_16_imagenet1k', pretrained=True)
        self.overlay_encoder = vision_encoder
        
        
        self.encoder = vision_encoder
        
        med_config = "/media/NAS/USERS/shahid/MultimodalAudioVisualModerator/models/xbm/configs/bert_config.json"
        vision_width=1024
        self.tokenizer = init_tokenizer()
        self.tokenizer.padding_side = "left"

        # many decoder-only models have no PAD token — set it to EOS
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel.from_pretrained(
            "bert-large-uncased", config=encoder_config, add_pooling_layer=False
        )
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))



        decoder_config = BertConfig.from_pretrained("bert-large-uncased")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder_config.encoder_width = vision_width  # must match the visual token dim
        self.text_decoder = BertLMHeadModel.from_pretrained(
            "bert-large-uncased", config=decoder_config
        )

        assert any("crossattention" in n for n, _ in self.text_decoder.named_modules()), \
            "No cross-attention blocks in decoder — it won't look at images."

        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        
        for param in self.text_decoder.parameters():
            param.requires_grad = False
        for layer in self.text_decoder.bert.encoder.layer:
            for p in layer.crossattention.parameters():
                p.requires_grad = True
        
        for param in self.text_decoder.cls.parameters():  # or .lm_head
            param.requires_grad = True


        # Unfreeze Q, K, V together (not just Q and V)
        for layer in self.text_decoder.bert.encoder.layer:
            layer.attention.self.query.weight.requires_grad = True
            layer.attention.self.query.bias.requires_grad = True
            layer.attention.self.key.weight.requires_grad = True      # ADD THIS
            layer.attention.self.key.bias.requires_grad = True        # ADD THIS
            layer.attention.self.value.weight.requires_grad = True
            layer.attention.self.value.bias.requires_grad = True

        
        decoder_config = BertConfig.from_pretrained("bert-large-uncased")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder_config.encoder_width = vision_width  # must match the visual token dim
        self.bert_encoder = BertLMHeadModel.from_pretrained(
            "bert-large-uncased", config=decoder_config
        )
        self.bert_encoder.resize_token_embeddings(len(self.tokenizer))
        self.bert_encoder.eval()
        
        text_width = self.text_encoder.config.hidden_size
        vocab_size = self.text_encoder.config.vocab_size
        self.text_embedding = nn.Embedding(vocab_size, dim)
        self.text_proj = nn.Linear(text_width, dim)
       

        self.momentum = 1.0
        self.prompt = "a picture of ",
        # self.sample = sample
        self.num_beams = 3
        self.max_decode_length = 77
        self.min_decode_length = 20
        self.top_p = 0.9
        self.repetition_penalty = 1.0
        self.use_cross_attn = True
        fixed_caps = True
        fixed_vit = True
        self.caption_context = torch.enable_grad
        self.vision_context =  torch.enable_grad
        # torch.no_grad if fixed_caps else 
        # torch.no_grad if fixed_vit else
        self.temperature = 1

        #Text model initialization ends

        # clip
        self.image_decoder = unet_decoder

        self.cls_token = nn.Parameter(torch.randn(14, dim))
        self.fc_norm = nn.LayerNorm(dim)
        self.fusion_attn = nn.MultiheadAttention(dim, 4, batch_first=True, device=self.device)
        
      
        self.cls_head = nn.Linear(dim , 1)    
        
        self.predictor = nn.Linear(dim,dim)
        # self.projector = nn.Linear(1536,768)
        self.projector_aug = nn.Linear(dim,dim)
        self.predictor_intra = nn.Linear(dim,dim)
        self.predictor_inter = nn.Linear(dim,dim)
        self.predictor_recon = nn.Linear(dim,dim)
        self.contrastive = Contrastive()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = BinaryFocalLoss()
        self.multi_cls_loss = nn.CrossEntropyLoss()
        self.token_simclr = TokenLoss(dim=dim)
        
        self.norm_pix_loss = True
        
        self._init_new_layers()
    

    def _init_new_layers(self):
        # Initialize classifier
        nn.init.xavier_uniform_(self.cls_head.weight)
        nn.init.constant_(self.cls_head.bias, 0)

        # self.copy_params()
    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def prepare_text_targets(self, heatmap, labels, max_length=77, visual_embeds=None, visual_mask=None):
        """
        Build LM training targets from the heatmap-driven captions.
        Returns:
            decoder_input_ids : [B, L-1]
            attention_mask    : [B, L-1]
            decoder_labels    : [B, L-1]   (-100 on pads)
            captions          : List[str]
        """
        device = labels.device
        captions, teacher_log_probs = self.generate_caption_from_heatmap(
            heatmap=heatmap, label=labels, threshold=0.45, diversity=0.65, n_variants=1, visual_embeds=visual_embeds, visual_mask=visual_mask, device=device
        )
        # print(captions)



        # Step 2: Use frozen LLM to normalize/semantic-refine captions
        # inputs = self.semantic_llm_tokenizer(
        #     captions, padding=True, truncation=True, return_tensors="pt"
        # ).to(device)

        # with torch.no_grad():
        #     semantic_emb = self.semantic_llm(**inputs).last_hidden_state.mean(dim=1)  # [B, D]

            
        tok = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.to(device)             # [B, L]
        attn_mask = tok.attention_mask.to(device)        # [B, L]

        # decoder inputs: shift right and place BOS at position 0
        # keep the common pattern: input_ids[:, :-1] as decoder inputs; labels[:, 1:]
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        if bos_id is None:
            # fall back to e.g. [PAD] if model lacks BOS; safe for BERT-LMHead
            bos_id = self.tokenizer.pad_token_id

        # prepend BOS, then cut to L
        bos_col = torch.full((input_ids.size(0), 1), bos_id, device=device, dtype=input_ids.dtype)
        dec_in = torch.cat([bos_col, input_ids[:, :-1]], dim=1)   # [B, L]

        # labels: ignore index on padding
        labels_full = input_ids.clone()
        labels_full[attn_mask == 0] = -100

        # train on next token: drop last column for inputs/attn, drop first for labels
        decoder_input_ids = dec_in[:, :-1]
        attention_mask = attn_mask[:, :-1]
        decoder_labels = labels_full[:, 1:]

        return decoder_input_ids, attention_mask, decoder_labels, captions,teacher_log_probs#, semantic_emb

    def _bucket(self,value: float, cuts=(0.25, 0.5, 0.75)) -> str:
        if value < cuts[0]: return "low"
        if value < cuts[1]: return "moderate"
        if value < cuts[2]: return "high"
        return "very high"

    @torch.no_grad()
    def _normalize_map(self,x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # x: [B, C, H, W] or [B, 1, H, W]
        x = x.mean(dim=1, keepdim=True)  # -> [B,1,H,W]
        mn, mx = x.amin(dim=(2,3), keepdim=True), x.amax(dim=(2,3), keepdim=True)
        return (x - mn) / (mx - mn + eps)

    
    @torch.no_grad()
    def unnormalize(self,t: torch.Tensor):
        """
        t: [C, H, W] or [B, C, H, W], normalized to ImageNet stats.
        Returns: unnormalized tensor in [0,1] range (clamped).
        """
                
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean, device=t.device).view(-1, 1, 1)
        std = torch.tensor(std, device=t.device).view(-1, 1, 1)
        return (t * std + mean).clamp(0, 1)

    def analyze_manipulation_regions(self, heatmap: torch.Tensor, threshold: float = 0.5, topk: int = 3):
        """
        Args:
            heatmap  : [B, C, H, W] (unnormalized OK)
            threshold: region considered 'manipulated' if avg >= threshold
            topk     : number of regions to keep in the summary
        Returns:
            List[dict] with keys:
            - regions: List[(name, avg, max, area_ratio, severity_str)]
            - top_regions: same as regions but topk by composite severity
            - confidence: float in [0,1]
            - overall_severity: str bucket by mean composite severity
        """
        B, C, H, W = heatmap.shape
        hm = self._normalize_map(heatmap)[:, 0]  # [B,H,W]

        # Simple geometric regions (can be swapped with landmarks later)
        regions_def = {
            'forehead': (slice(0, H//4), slice(W//4, 3*W//4)),
            'eyes':     (slice(H//4, H//2), slice(W//5, 4*W//5)),
            'nose':     (slice(H//3, 2*H//3), slice(2*W//5, 3*W//5)),
            'mouth':    (slice(H//2, 3*H//4), slice(W//3, 2*W//3)),
            'cheeks':   (slice(H//3, 2*H//3), slice(0, W)),
            'chin':     (slice(3*H//4, H), slice(W//3, 2*W//3)),
            'overall_face': (slice(H//6, 5*H//6), slice(W//6, 5*W//6)),
        }
        area_total = float(H * W)

        out = []
        for b in range(B):
            entries = []
            comp_scores = []
            for name, (hs, ws) in regions_def.items():
                r = hm[b, hs, ws]
                avg = float(r.mean().item())
                mx  = float(r.max().item())
                area_ratio = float(r.numel() / area_total)
                # composite favors average (texture inconsistency) with a bit of max (localized spikes)
                composite = 0.7 * avg + 0.3 * mx
                severity = self._bucket(composite)  # "low/moderate/high/very high"
                entries.append((name, avg, mx, area_ratio, severity, composite))
                comp_scores.append(composite)

            # sort by composite desc and keep topk
            entries.sort(key=lambda t: t[-1], reverse=True)
            top = entries[:topk]
            mean_comp = float(np.mean(comp_scores))
            overall_severity = self._bucket(mean_comp)

            # confidence: higher if distribution is peaky and mean is sizable
            # (soft heuristic: mean * (max - second_max))
            peak_gap = (entries[0][-1] - entries[1][-1]) if len(entries) > 1 else entries[0][-1]
            confidence = max(0.0, min(1.0, mean_comp * (0.5 + peak_gap)))

            # strip composite from public dicts
            def _strip(ls):
                return [(n, a, m, ar, s) for (n, a, m, ar, s, _) in ls]

            out.append({
                "regions": _strip(entries),
                "top_regions": _strip(top),
                "confidence": confidence,
                "overall_severity": overall_severity
            })
        return out
    
    def generate_caption_from_heatmap(
        self,
        heatmap: torch.Tensor,
        label: torch.Tensor,
        threshold: float = 0.45,
        diversity: float = 0.6,
        n_variants: int = 1,
        visual_embeds=None,
        visual_mask=None,
        device=None,
    ) -> List[str]:
        B = heatmap.size(0)
        analysis = self.analyze_manipulation_regions(heatmap, threshold=threshold, topk=3)
        captions_seed = []

        artifact_syns = ["artifacts","forgery cues","editing traces","synthetic residues"]
        detected_syns = ["detected","observed","found","identified"]
        texture_syns  = ["texture","surface consistency","skin pattern","micro-texture"]
        boundary_syns = ["boundaries","edges","contours","facial outlines"]
        lighting_syns = ["lighting","illumination","shading","color gradient"]
        hedge_syns    = ["with high confidence","with moderate confidence","likely","possibly"]
        pick = lambda lst: random.choice(lst)

        for b in range(B):
            is_fake = bool(label[b].item() == 1)
            info = analysis[b]
            top_regs = info["top_regions"]
            sev = info["overall_severity"]
            conf = info["confidence"]

            if top_regs:
                rname = _fmt_regions(top_regs, limit=3)
                avg_hint = f"{top_regs[0][1]:.2f}"
                mx_hint  = f"{top_regs[0][2]:.2f}"
                quant_snip = f"(avg={avg_hint}, max={mx_hint})"
            else:
                rname, quant_snip = "the facial area", ""

            kernel = _severity_phrase(sev, is_fake)
            maybe_hedge = "" if conf > 0.75 else f" ({pick(hedge_syns)})"

            A, D, T, Bn, L = artifact_syns, detected_syns, texture_syns, boundary_syns, lighting_syns
            if is_fake:
                templates = [
                    f"The image seems fake. face shows {pick(A)} {pick(D)} around {rname} {quant_snip}, revealing {kernel}{maybe_hedge}.",
                    f"This image shows fake face. {pick(A).capitalize()} are {pick(D)} in {rname} {quant_snip}; variations in {pick(T)} and {pick(Bn)} indicate {kernel}{maybe_hedge}.",
                    f"The image is fake. Signs of manipulation appear near {rname}: irregular {pick(T)} and {pick(L)} patterns mark {kernel}{maybe_hedge}.",
                    f"The image shows fake facial features. {pick(A).capitalize()} concentrated in {rname} {quant_snip} correspond to {kernel}{maybe_hedge}.",
                    f"The face image shows forgeries, and shows a fake forged iamge. Localized inconsistencies in {pick(T)} and {pick(L)} across {rname} point to {kernel}{maybe_hedge}.",
                ]
            else:
                templates = [
                    f"This appears real: {rname} remains stable {quant_snip}, with consistent {pick(T)} and {pick(L)} across the face{maybe_hedge}.",
                    f"The image is real. No strong evidence of tampering; {rname} shows natural {pick(T)} and coherent {pick(Bn)} {quant_snip}{maybe_hedge}.",
                    f"This is real image. Facial integrity preserved — {pick(T)} uniform, {pick(L)} coherent, and no high-score regions found{maybe_hedge}.",
                    f"The image shows real face without any artifacts. Authentic cues dominate: no abnormal {pick(Bn)} or {pick(T)} beyond thresholds {quant_snip}{maybe_hedge}.",
                    f"The image is real. Reconstruction yields low manipulation confidence; all examined zones show realistic {pick(T)} and {pick(L)}{maybe_hedge}.",
                ]

            # if is_fake:
            #     templates = [
            #         f"face shows {pick(A)} {pick(D)} around {rname} {quant_snip}, revealing {kernel}{maybe_hedge}.",
            #         f"{pick(A).capitalize()} are {pick(D)} in {rname} {quant_snip}; variations in {pick(T)} and {pick(Bn)} indicate {kernel}{maybe_hedge}.",
            #         f"Signs of manipulation appear near {rname}: irregular {pick(T)} and {pick(L)} patterns mark {kernel}{maybe_hedge}.",
            #         f"{pick(A).capitalize()} concentrated in {rname} {quant_snip} correspond to {kernel}{maybe_hedge}.",
            #         f"Localized inconsistencies in {pick(T)} and {pick(L)} across {rname} point to {kernel}{maybe_hedge}.",
            #     ]
            # else:
            #     templates = [
            #         f" {rname} remains stable {quant_snip}, with consistent {pick(T)} and {pick(L)} across the face{maybe_hedge}.",
            #         f"No strong evidence of tampering; {rname} shows natural {pick(T)} and coherent {pick(Bn)} {quant_snip}{maybe_hedge}.",
            #         f"Facial integrity preserved — {pick(T)} uniform, {pick(L)} coherent, and no high-score regions found{maybe_hedge}.",
            #         f"Authentic cues dominate: no abnormal {pick(Bn)} or {pick(T)} beyond thresholds {quant_snip}{maybe_hedge}.",
            #         f"Reconstruction yields low manipulation confidence; all examined zones show realistic {pick(T)} and {pick(L)}{maybe_hedge}.",
            #     ]

            # lexical/structural variety
            variants = []
            for _ in range(n_variants):
                s = random.choice(templates)
                if random.random() < diversity * 0.5:
                    s = s.replace("shows", random.choice(["reveals","contains","exhibits"]))
                if random.random() < diversity * 0.3:
                    s = s.replace("point to", random.choice(["suggest","indicate","are consistent with"]))
                if random.random() < diversity * 0.4:
                    s = s.replace("appear", random.choice(["emerge","become visible","can be seen"]))
                if random.random() < diversity * 0.25:
                    s = s.replace("localized inconsistencies", "minor localized distortions")
                if random.random() < diversity * 0.2:
                    s = s.replace("no strong evidence", "no convincing sign")
                if random.random() < diversity * 0.4:
                    segs = s.split(";")
                    random.shuffle(segs)
                    s = "; ".join(segs)
                variants.append(s.strip())

                captions_seed.append(random.choice(variants))
                # captions_seed.append("Real" if not is_fake else "Fake")
                
        # return captions_seed, torch.randn(512)
            

        if visual_embeds is not None and hasattr(self, "bert_encoder"):
            refined_captions = []
            teacher_probs_list = []

            for idx, cap in enumerate(captions_seed):
                is_fake = bool(label[idx].item() == 1)
                argument = "Fake" if is_fake else "Real"

                # Convert heatmap to PIL image
                img_np = (
                    heatmap[idx].detach().cpu().clamp(0, 1)
                    .permute(1, 2, 0).numpy() * 255
                ).astype(np.uint8)
                img_pil = Image.fromarray(img_np)

                # Construct prompt
                prompt = (
                    f"Improve the clarity and grammar of the following analytical caption for a {argument.lower()} image, "
                    "without adding any new scene description or background details:\n"
                    f"'{cap}'\nRefined version:"
                )
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.bert_encoder.device) for k, v in inputs.items()}
                inputs.pop("token_type_ids", None)  # decoder usually doesn't use segments

                # Build encoder states from visual tokens for this sample
                # visual_embeds: [B, Nv, vision_width]; visual_mask: [B, Nv]
                enc_hid = visual_embeds[idx].to(self.bert_encoder.device)          # [Nv, vision_width]
                enc_hid = enc_hid.unsqueeze(0)                                     # [1, Nv, vision_width]

                if visual_mask is not None:
                    enc_mask = visual_mask[idx].to(self.bert_encoder.device)       # [Nv]
                    enc_mask = enc_mask.unsqueeze(0)                                # [1, Nv]
                else:
                    enc_mask = torch.ones(enc_hid.shape[:2], dtype=torch.long, device=enc_hid.device)

                # (Optional) project vision width -> BERT hidden size if needed
                if getattr(self, "vis2txt", None) is not None:
                    enc_hid = self.vis2txt(enc_hid)                                # [1, Nv, hidden_size]

                with torch.no_grad():
                    outputs = self.bert_encoder(
                        input_ids=inputs["input_ids"],                  # [1, Lt]
                        attention_mask=inputs.get("attention_mask"),    # [1, Lt]
                        encoder_hidden_states=enc_hid,                  # [1, Nv, hidden]
                        encoder_attention_mask=enc_mask,                # [1, Nv]
                        output_hidden_states=False,
                        output_attentions=False,
                    )
                    teacher_logits = outputs.logits                     # [1, Lt, vocab]
                    temperature = getattr(self, "teacher_temperature", 1.0)
                    probs = torch.softmax(teacher_logits / temperature, dim=-1)
                    teacher_probs_list.append(probs)

                # (Optional) If you later want text refinement, decode here
                # refined = self.blip2_processor.batch_decode(
                #     torch.argmax(probs, dim=-1), skip_special_tokens=True
                # )[0].strip()
                # refined_captions.append(refined)

            # Stack all teacher probs (batch-level)
            padded = pad_sequence(
                [p.squeeze(0) for p in teacher_probs_list],  # each [seq_len, vocab]
                batch_first=True,  # -> [B, max_len, vocab]
                padding_value=0.0
            )
            teacher_probs = padded
            # print(teacher_probs.shape)
            # teacher_probs = (
            #         torch.cat(teacher_probs_list, dim=0)
            #         if len(teacher_probs_list) > 0
            #         else None
            #     )

        return captions_seed, teacher_probs

        # Fallback — if BLIP2 not present
        return captions_seed, None

        # # return captions_seed
        # # return captions_seed
        # # ----- BLIP refinement (batch) -----
        # if visual_embeds is not None and hasattr(self, "blip2_model"):
        #     # Convert overlay to a PIL image
        #     refined_captions = []
        #     for idx, cap in enumerate(captions_seed):
        #         is_fake = bool(label[idx].item() == 1)
        #         aurgument = "Fake" if is_fake else "Real"
        #         # Get image from overlay tensor
        #         # img_np = (heatmap[idx].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        #         # img_np = self.unnormalize(heatmap[idx])
        #         img_np = (heatmap[idx].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
        #         img_pil = Image.fromarray(img_np)
               
        #         # Feed to BLIP2
        #         # prompt = f"Question: refine the description ``{cap}'' of a {aurgument} image, Answer:"
        #         # prompt = f"Refine this caption for a {aurgument.lower()} image: {cap}"
        #         prompt = (
        #             f"Improve the clarity and grammar of the following analytical caption for a {aurgument.lower()} image, "
        #             "without adding any new scene description or background details:\n"
        #             f"'{cap}'\nRefined version:"
        #         )
        #         inputs = self.blip2_processor(images=img_pil, text=prompt, return_tensors="pt").to(self.blip2_model.device)

        #         with torch.no_grad():
        #             outputs = self.blip2_model(**inputs, output_hidden_states=False, output_attentions=False)
        #             teacher_logits = outputs.logits  # shape: [B, seq_len, vocab_size]
        #             teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
        #         # out = self.blip2_model.generate(**inputs,
        #         #         min_length=90,
        #         #         max_new_tokens=150,
        #         #         temperature=0.7,
        #         #         do_sample=True,
        #         #         top_p=0.9,
        #         #         length_penalty=1.2)
        #         # # inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)
        #         # refined = self.blip2_processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        #         # refined_captions.append(refined)
           

        # return captions_seed, teacher_probs
        # # return captions_seed

        
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def infer(self, image, class_label=None, ret_caption=False, ret_attention=False):
        # 1. Generate Image Embeddings
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # 2. Generate Caption
        with torch.no_grad():
            image_embeds_beam = image_embeds.detach().repeat_interleave(self.num_beams, dim=0)
            image_atts_beam = torch.ones(image_embeds_beam.size()[:-1], dtype=torch.long).to(image.device)
            decoder_kwargs_beam = {
                "encoder_hidden_states": image_embeds_beam,
                "encoder_attention_mask": image_atts_beam,
            }
            prompt = [self.prompt] * image.size(0)
            input_ids_m = self.tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(image.device)
            input_ids_m[:, 0] = self.tokenizer.bos_token_id
            input_ids_m = input_ids_m[:, :-1]
            # beam search
            decoder_output = self.text_decoder.generate(
                input_ids=input_ids_m,
                max_length=self.max_decode_length,
                min_length=self.min_decode_length,
                num_beams=self.num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=self.repetition_penalty,
                **decoder_kwargs_beam,
            )
            captions = []
            for output in decoder_output:
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                # captions.append(caption[len(self.prompt) :])
                captions.append(caption)

        # 3. Encode Caption (and Image Embedding)
        caption_text = self.tokenizer(
            captions,
            # padding="max_length",
            padding=True, 
            truncation=True,
            max_length=self.max_decode_length,
            return_tensors="pt",
        ).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        cross_attn_kwargs = {"mode": "text"} if not self.use_cross_attn else model_kwargs
        caption_text_output = self.text_encoder(
            caption_text.input_ids,
            attention_mask=caption_text.attention_mask,
            return_dict=True,
            output_attentions=True,
            **cross_attn_kwargs,
        )
        text_feat = F.normalize(self.text_proj(caption_text_output.last_hidden_state[:, 0, :]), dim=-1)

        # 5. Predict Final Labels
        logit = self.classifier(text_feat)
        if class_label is not None:
            loss_cls = F.cross_entropy(logit, class_label)
            acc = accuracy(logit.detach(), class_label)[0]
        else:
            loss_cls, acc = None, None

        if ret_attention:
            return logit, captions, caption_text_output
        if ret_caption:
            return loss_cls, acc, captions
        else:
            return loss_cls, acc

    @torch.no_grad()
    def generate(self, image, prompt="a picture of", ret_attention=False):
        # 1. Generate Image Embeddings
        image_embeds = self.visual_encoder(image)
        # image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        if not self.sample:
            image_embeds_beam = image_embeds.detach().repeat_interleave(self.num_beams, dim=0)
            image_atts_beam = torch.ones(image_embeds_beam.size()[:-1], dtype=torch.long).to(image.device)
            decoder_kwargs = {"encoder_hidden_states": image_embeds_beam, "encoder_attention_mask": image_atts_beam}
        else:
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            decoder_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        # 2. Decode Caption from Image Embeddings
        prompt = [self.prompt] * image.size(0)
        decoder_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        # decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)

        if self.sample:
            # nucleus sampling
            decoder_outputs = self.text_decoder.generate(
                input_ids=decoder_input_ids,
                max_length=self.max_decode_length,
                min_length=self.min_decode_length,
                do_sample=True,
                top_p=self.top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                output_attentions=True,
                return_dict_in_generate=True,
                **decoder_kwargs,
            )
        else:
            # beam search
            decoder_outputs = self.text_decoder.generate(
                input_ids=decoder_input_ids,
                max_length=self.max_decode_length,
                min_length=self.min_decode_length,
                num_beams=self.num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=self.repetition_penalty,
                return_dict_in_generate=True,
                output_attentions=True,
                output_hidden_states=True,
                **decoder_kwargs,
            )
        captions = []
        for output in decoder_outputs.sequences:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption)
        if ret_attention:
            return captions, decoder_outputs
        else:
            return captions


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


    def forward_mse_loss(self, target, pred):
        pred = F.interpolate(pred, size=(target.shape[2], target.shape[3]), 
                                mode='bilinear', align_corners=False)
        # print(pred.shape, target.shape)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(var + 1e-6)
        
        loss = (pred - target).pow(2)
        loss = loss.mean()
    
        return loss

        
    # def forward_mse_loss(self, target, pred):
    # # Ensure pred shape matches target shape
    #     if pred.shape != target.shape:
    #         # Check if it's a spatial dimension mismatch (common case)
    #         if len(pred.shape) == 4 and len(target.shape) == 4:  # [B, C, H, W]
    #             target_h, target_w = target.shape[-2:]
    #             pred = F.interpolate(pred, size=(target_h, target_w), 
    #                             mode='bilinear', align_corners=False)
    #         elif len(pred.shape) == 3 and len(target.shape) == 3:  # [B, H, W] or [C, H, W]
    #             target_h, target_w = target.shape[-2:]
    #             pred = F.interpolate(pred.unsqueeze(1), size=(target_h, target_w),
    #                             mode='bilinear', align_corners=False).squeeze(1)
    #         elif len(pred.shape) == 2 and len(target.shape) == 2:  # [H, W]
    #             pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), 
    #                             size=target.shape, mode='bilinear', 
    #                             align_corners=False).squeeze(0).squeeze(0)
    #         else:
    #             # For other dimension mismatches, try adaptive pooling or error
    #             try:
    #                 if pred.numel() > target.numel():
    #                     # Downsample pred to match target
    #                     pred = F.adaptive_avg_pool2d(pred, target.shape[-2:])
    #                 else:
    #                     # Upsample pred to match target  
    #                     pred = F.interpolate(pred, size=target.shape[-2:], 
    #                                     mode='bilinear', align_corners=False)
    #             except Exception as e:
    #                 raise ValueError(f"Cannot match pred shape {pred.shape} to target shape {target.shape}: {e}")
        
    #     # Apply pixel normalization if enabled
    #     if self.norm_pix_loss:
    #         mean = target.mean(dim=-1, keepdim=True)
    #         var = target.var(dim=-1, keepdim=True)
    #         target = (target - mean) / torch.sqrt(var + 1e-6)
            
    #         # Apply same normalization to pred for consistency
    #         pred_mean = pred.mean(dim=-1, keepdim=True)
    #         pred_var = pred.var(dim=-1, keepdim=True)
    #         pred = (pred - pred_mean) / torch.sqrt(pred_var + 1e-6)
        
    #     # Compute MSE loss
    #     loss = (pred - target).pow(2)
    #     loss = loss.mean()

    #     return loss


    def create_gt_heatmap(self, img, score, thr: float = 0.30, alpha: float = 0.4, return_heat: bool = True):
      
        """
        img   : torch.Tensor [B,3,H,W] in [0,1] or [0,255]
        score : torch.Tensor [B,1,H,W] or [B,C,H,W] (logits/scores/mask); any real range
        thr   : threshold in 0..1 on the *normalized* heat
        alpha : overlay strength
        return: overlay_rgb [B,3,H,W] float in 0..1,
                bin_mask   [B,1,H,W] float in 0..1,
                heat_norm  [B,1,H,W] float in 0..1  (if return_heat)
        """
        import numpy as np, cv2, torch
        B = img.shape[0]
        # to CPU numpy, clamp and scale
        img_t = img.detach().float().cpu()
        if img_t.max() > 1.0: img_t = img_t 
        img_np = (img_t.clamp(0,1) * 255).permute(0,2,3,1).numpy().astype(np.uint8)  # B,H,W,3

        s = score.detach().float().cpu()
        if s.ndim == 3:  # [B,H,W] -> [B,1,H,W]
            s = s.unsqueeze(1)
        if s.size(1) > 1:
            s = s.abs().mean(dim=1, keepdim=True)  # reduce to 1-channel

        # normalize per-sample ignoring constant maps
        s = s - s.amin(dim=(2,3), keepdim=True)
        denom = s.amax(dim=(2,3), keepdim=True) + 1e-6
        heat = (s / denom).clamp(0,1)    # [B,1,H,W] in 0..1

        overlays, bin_masks, heat_list = [], [], []
        for b in range(B):
            h8   = (heat[b,0].numpy() * 255).astype(np.uint8)                # H,W
            cm   = cv2.applyColorMap(h8, cv2.COLORMAP_JET)                   # H,W,3 BGR
            cm   = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)                       # to RGB
            face = img_np[b]                                                 # H,W,3 RGB uint8
            ov   = cv2.addWeighted(face, 1.0 - alpha, cm, alpha, 0)

            overlays.append(torch.from_numpy(ov).permute(2,0,1).float() )
            bin_masks.append(torch.from_numpy((h8 > int(thr*255)).astype(np.float32))[None])
            heat_list.append(torch.from_numpy(h8)[None].float() )

        overlay_rgb = torch.stack(overlays, 0)
        mask_bin    = torch.stack(bin_masks, 0)
        heat_norm   = torch.stack(heat_list, 0)
        return (overlay_rgb, mask_bin, heat_norm) if return_heat else (overlay_rgb, mask_bin)

    @torch.no_grad()
    def gradcam_overlay(
        self,
        input_images: torch.Tensor,   # CLIP-normalized [B,3,H,W]
        heat_src: torch.Tensor,       # [B,C,H',W'] (scores/logits)
        alpha: float = 0.45,
        assume_sigmoid: bool = False,
        per_sample_norm: bool = True,
        gamma: float = 1.0,
        gate_by_heat: bool = False,   # optional, see below
    ):
        B, _, H, W = input_images.shape

        # 0) **DENORMALIZE** CLIP -> RGB [0,1] for visualization
        
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std  = [0.26862954, 0.26130258, 0.27577711]
        clip_mean= torch.tensor(clip_mean).view(1,3,1,1).to(input_images.device)
        clip_std= torch.tensor(clip_std).view(1,3,1,1).to(input_images.device)
        img = (input_images * clip_std) + clip_mean
        img = img.clamp(0, 1)

        # 1) Resize heat to image size
        heat = F.interpolate(heat_src.float(), size=(H, W), mode='bilinear', align_corners=False)
        if assume_sigmoid:
            heat = heat.sigmoid()
        if heat.size(1) > 1:
            heat = heat.mean(dim=1, keepdim=True)    # [B,1,H,W]

        # 2) Normalize heat 0..1
        if per_sample_norm:
            hmin = heat.amin(dim=(2,3), keepdim=True)
            hmax = heat.amax(dim=(2,3), keepdim=True)
            heat = (heat - hmin) / (hmax - hmin + 1e-6)
        else:
            heat = heat.clamp(0, 1)
        if gamma != 1.0:
            heat = heat.pow(gamma)
        heat_gray = heat  # [B,1,H,W]

        # 3) JET colormap (Torch)
        x = heat_gray
        r = (1.5 - torch.abs(4*x - 3)).clamp(0,1)
        g = (1.5 - torch.abs(4*x - 2)).clamp(0,1)
        b = (1.5 - torch.abs(4*x - 1)).clamp(0,1)
        heat_color = torch.cat([r, g, b], dim=1)  # [B,3,H,W]

        # 4) Blend (classic or gated)
        if gate_by_heat:
            # background stays mostly original; hot regions get colored
            overlay = (1 - alpha*heat_gray) * img + (alpha*heat_gray) * heat_color
        else:
            overlay = (1 - alpha) * img + alpha * heat_color
        overlay = overlay.clamp(0, 1)

        return overlay, heat_gray, heat_color

        
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
      
        text_aug = text
      
        input_images = images
        if self.training:
            input_images = torch.concat([images, images_aug], dim=0)

        if self.training:
            input_images = torch.concat([images, images_aug], dim=0)
            labels_expanded = torch.concat([labels, torch.ones_like(labels)], dim=0)
        else:
            labels_expanded = labels
       

        # dinov3 code
        tokens, embeddings = self.encoder(input_images)

        mse_loss_images = 0
        mse_loss_images_black = 0
        
        embeddings = embeddings[:, 1:, :]
        image_patch_embed, image_recon = self.image_decoder(embeddings)
       
            # print(video_recon.shape, multi_label.shape)
            
        if self.training:
            # 
            multi_labels = torch.concat([torch.zeros_like(multi_label), multi_label], dim=0)
            mse_loss_images = self.forward_mse_loss(multi_labels, image_recon)

        
        image_recon_vis = F.interpolate(
                        image_recon,
                        size=(input_images.shape[2], input_images.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
        # overlay, overlay_mask = self.create_gt_heatmap(img=input_images.clone().detach(),pix_diff=image_recon_vis)
                
        # overlay, overlay_mask, heat = self.create_gt_heatmap(
        #     img=input_images, score=image_recon_vis, thr=0.30, alpha=0.2, return_heat=True
        # )

        
        overlay, heat_gray, heat_color = self.gradcam_overlay(
            input_images=input_images,
            heat_src=image_recon_vis.clamp(0,1)/255.0,     # your recon / attention map
            alpha=0.5,
            assume_sigmoid=False,     # True if logits
            per_sample_norm=True,
            gamma=1.0
        )
       
           # ← only color where heat>0
        overlay, overlay_mask = overlay.to(input_images.device), heat_gray.to(input_images.device)
        
                    
        tokens_recon, embeddings_recon = self.overlay_encoder(overlay)
        

        embeddings_recon = embeddings_recon[:, 1:, :]
        embeddings_m = torch.ones(
            embeddings_recon.size()[:-1], 
            dtype=torch.long
        ).to(input_images.device)
    

        if self.training:

            with torch.no_grad():
                image_mask_vis = F.interpolate(
                        multi_labels,
                        size=(input_images.shape[2], input_images.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
                # overlay_ori, overlay_mask_ori, heat = self.create_gt_heatmap(img=input_images.clone().detach(),score=image_mask_vis, return_heat=True)
                        
                overlay_ori, heat_gray, heat_color = self.gradcam_overlay(
                    input_images=input_images.clone().detach(),
                    heat_src=image_mask_vis,     # your recon / attention map
                    alpha=0.8,
                    assume_sigmoid=False,     # True if logits
                    per_sample_norm=True,
                    gamma=1.0                 # try 0.7 or 0.5 for punchier highlights
                )
                
                overlay_ori, overlay_mask_ori = overlay_ori.to(input_images.device), heat_gray.to(input_images.device)
                tokens_recon_ori, embeddings_ori = self.encoder(overlay_ori)
                    
                # overlay_ori = overlay.clamp(0, 1)
                embeddings_ori_m = torch.ones(
                    embeddings_ori.size()[:-1], 
                    dtype=torch.long
                ).to(input_images.device)
        else:
            overlay_ori = overlay
            embeddings_ori = embeddings_recon
            embeddings_ori_m = torch.ones(
                embeddings_ori.size()[:-1], 
                dtype=torch.long
            ).to(input_images.device)
       






        generated_captions = ["No generated captions available"]
  
        # if self.training:
            # ------- TRAINING MODE: Generate captions from heatmap -------
            
            # Generate captions based on heatmap analysis
        decoder_input_ids, decoder_attention_mask, decoder_labels, generated_captions, teacher_log_probs = \
            self.prepare_text_targets(overlay_ori, labels_expanded, max_length=self.max_decode_length, visual_embeds=embeddings_ori, visual_mask=embeddings_ori_m)
        with self.caption_context():
        # Run decoder with teacher forcing on generated captions
            decoder_outputs = self.text_decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=embeddings_recon,
                encoder_attention_mask=embeddings_m,
                labels=decoder_labels,
                return_dict=True,
            )
            tau = self.temperature 
            # token_mask = nn.functional.gumbel_softmax(decoder_outputs.logits, tau=tau, hard=True, dim=-1)
            # indices = torch.tensor(range(0, token_mask.size(-1))).to(token_mask.device)
            # decoded_tokens = (token_mask * indices).sum(-1)
            student_probs = decoder_outputs.logits
             # Stack all teacher probs (batch-level)
            student_probs = pad_sequence([p.squeeze(0) for p in student_probs], batch_first=True, padding_value=0.0)

            token_mask = F.gumbel_softmax(decoder_outputs.logits, tau=self.temperature, hard=True, dim=-1)
            decoded_tokens = token_mask.argmax(dim=-1).long()   # <- convert one-hot -> indices
            # decoded_tokens = torch.multinomial(probs, 1).squeeze(-1).long()
            assert decoded_tokens.dtype in (torch.int32, torch.int64), f"Bad dtype: {decoded_tokens.dtype}"

            captions = []
            for output in decoded_tokens.detach():
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                captions.append(caption)
            loss_lm = decoder_outputs.loss

        
        # 4. For inference, use generate() properly
        with torch.no_grad():
            gen_sequences = self.text_decoder.generate(
                input_ids=decoder_input_ids,
                max_length=self.max_decode_length,
                min_length=self.min_decode_length,
                num_beams=self.num_beams,
                do_sample=False,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=self.repetition_penalty,
                encoder_hidden_states=embeddings_recon,
                encoder_attention_mask=embeddings_m,
            )
            
            # Convert to readable text
            captions_readable = self.tokenizer.batch_decode(
                gen_sequences,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Clean up (optional)
            captions_readable = [cap.strip() for cap in captions_readable]
        # 4. Encode Caption (and Image Embedding)
        caption_text = self.tokenizer(
            captions_readable,
            padding="max_length",
            truncation=True,
            max_length=decoded_tokens.size(1),
            return_tensors="pt",
        ).to(images.device)

        model_kwargs = {"encoder_hidden_states": embeddings_recon, "encoder_attention_mask": embeddings_m}
        cross_attn_kwargs = {"mode": "text"} if not self.use_cross_attn else model_kwargs


        caption_text_output = self.text_encoder(
            decoded_tokens,
            attention_mask=caption_text.attention_mask,
            return_dict=True,
            output_attentions=True,
            **cross_attn_kwargs,
        )
        text_feat = F.normalize(self.text_proj(caption_text_output.last_hidden_state[:, 0, :]), dim=-1)
       
        if not self.pretraining: 
            # emb = self.text_embedding(decoded_tokens.long())
            # input_ids_encoded = self.text_proj(emb)
            # print(input_ids.shape)
            pooled, af_tok,a_tok, vf_tok, v_tok,tf_tok, t_tok, if_tok, i_tok = self.encoder_bottleneck(
                    audio=None,
                    video=None,
                    text_tokens=text_feat,
                    image=image_recon
                )
            af_tok_cls,a_tok_cls, vf_tok_cls, v_tok_cls,tf_tok_cls, t_tok_cls, if_tok_cls, i_tok_cls = pooled.unbind(dim=1)

        if not self.pretraining:
                
            va_out, _ = self.fusion_attn(pooled, pooled, pooled)
            f_out = self._safe_mean(va_out)
            # f_out = tokens
            assert torch.isfinite(f_out).all(), "f_out has NaNs"
            classes_logits = self.fc_norm(f_out)
            # classes_logits = self.classes_stream(classes_logits)
            # projection_out = self.proj_head(classes_logits)
            logits_cls = self.cls_head(classes_logits)
        else:
            logits_cls = torch.randn((images.size(0), 1), device=images.device)

        # ---------------- contrastive losses ----------------
        losses: Dict[str, Dict[str, torch.Tensor]] = {
            "intra": {},  # per-modality SimCLR
            "inter": {},  # cross-modal contrastive
            "lipsyncLoss":{},
            "reconstruction":{},
            "llm_loss":{},
            "kl_loss":{},
            "moe_loss":{},
            "cls_loss":{}, 
            "llm_samantic_loss":{}
        }

        
        losses["reconstruction"]["loss"] = mse_loss_images 
        student_emb = caption_text_output.last_hidden_state.mean(dim=1)

        # Compute semantic consistency with frozen LLM embedding
        # loss_semantic = self.semantic_loss(student_emb, semantic_emb)
        # losses["llm_samantic_loss"]["semantic"] = loss_semantic
        if self.training:

            losses["kl_loss"]["loss"] = F.kl_div(teacher_log_probs, student_probs, reduction='batchmean')
            # clamp to avoid log(0) or division by zero
            teacher_log_probs = torch.clamp(teacher_log_probs, min=-20.0, max=5.0)  # keep log-probs bounded
            student_probs = torch.clamp(student_probs, min=1e-8, max=1.0)

            # replace NaN / inf with 0 before KL
            teacher_log_probs = torch.nan_to_num(teacher_log_probs, nan=0.0, posinf=0.0, neginf=0.0)
            student_probs = torch.nan_to_num(student_probs, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                kl_val = F.kl_div(
                    teacher_log_probs, student_probs, reduction="batchmean"
                )
                if torch.isnan(kl_val) or torch.isinf(kl_val):
                    print("⚠️ KL overflow detected — skipping this batch.")
                    kl_val = torch.tensor(0.0, device=teacher_log_probs.device)
            except Exception as e:
                print("⚠️ KL computation failed:", e)
                kl_val = torch.tensor(0.0, device=teacher_log_probs.device)

            losses["kl_loss"]["loss"] = kl_val

        bce_weight = 0.8
        multi_weight = 0.2

        if not self.pretraining:
                
        # classes_logits = self._safe_mean(classes_logits)
            logits_cls = logits_cls.view(-1)
            # losses["cls_loss"]["loss"] = bce_weight * self.cls_loss(logits_cls.view(-1), labels) + multi_weight * self.multi_cls_loss(classes_logits, multi_label.long())
            losses["cls_loss"]["loss"] = self.cls_loss(logits_cls, labels_expanded) 
            
            
            # losses["cls_loss"]["llm_loss"] = loss_lm
            losses["llm_loss"]["lm"] = loss_lm
            
            ##################################
            #       reconstruction loses    ##
            ##################################
       
        return logits_cls, losses, labels_expanded, image_recon,generated_captions, captions,captions_readable, overlay, overlay_ori




# model = MMModerator()
# model(mfcc=torch.randn((24,40,63)), video=torch.randn((24,50,3,224,224)), mfcc_aug=torch.randn((24,40,63)), video_aug=torch.randn((24,50,3,224,224)))



# Custom module to represent the residual using SVD components
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of top singular values to exclude

        # Original weights (fixed)
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        else:
            return self.weight_main

    def forward(self, x):
        if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
            # Reconstruct the residual weight
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Total weight is the fixed main weight plus the residual
            weight = self.weight_main + residual_weight
        else:
            # If residual components are not set, use only the main weight
            weight = self.weight_main

        return F.linear(x, weight, self.bias)
    
    def compute_orthogonal_loss(self):
        if self.S_residual is not None:
            # According to the properties of orthogonal matrices: A^TA = I
            UUT = torch.cat((self.U_r, self.U_residual), dim=1) @ torch.cat((self.U_r, self.U_residual), dim=1).t()
            VVT = torch.cat((self.V_r, self.V_residual), dim=0) @ torch.cat((self.V_r, self.V_residual), dim=0).t()
            # print(self.U_r.size(), self.U_residual.size())  # torch.Size([1024, 1023]) torch.Size([1024, 1])
            # print(self.V_r.size(), self.V_residual.size())  # torch.Size([1023, 1024]) torch.Size([1, 1024])
            # UUT = self.U_residual @ self.U_residual.t()
            # VVT = self.V_residual @ self.V_residual.t()
            
            # Construct an identity matrix
            UUT_identity = torch.eye(UUT.size(0), device=UUT.device)
            VVT_identity = torch.eye(VVT.size(0), device=VVT.device)
            
            # Using frobenius norm to compute loss
            loss = 0.5 * torch.norm(UUT - UUT_identity, p='fro') + 0.5 * torch.norm(VVT - VVT_identity, p='fro')
        else:
            loss = 0.0
            
        return loss

    def compute_keepsv_loss(self):
        if (self.S_residual is not None) and (self.weight_original_fnorm is not None):
            # Total current weight is the fixed main weight plus the residual
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Frobenius norm of current weight
            weight_current_fnorm = torch.norm(weight_current, p='fro')
            
            loss = torch.abs(weight_current_fnorm ** 2 - self.weight_original_fnorm ** 2)
            # loss = torch.abs(weight_current_fnorm ** 2 + 0.01 * self.weight_main_fnorm ** 2 - 1.01 * self.weight_original_fnorm ** 2)
        else:
            loss = 0.0
        
        return loss
    
    def compute_fn_loss(self):
        if (self.S_residual is not None):
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight_current_fnorm = torch.norm(weight_current, p='fro')
            
            loss = weight_current_fnorm ** 2
        else:
            loss = 0.0
        
        return loss





class EnhancedRegionAnalyzer:
    """
    More sophisticated region analysis using facial landmarks or segmentation.
    """
    
    def __init__(self, use_landmarks=False):
        self.use_landmarks = use_landmarks
        if use_landmarks:
            try:
                import dlib
                import cv2
                self.detector = dlib.get_frontal_face_detector()
                # Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
                self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            except:
                print("Warning: dlib not available, using grid-based regions")
                self.use_landmarks = False
    
    def get_landmark_regions(self, image):
        """
        Get facial regions based on 68-point landmarks.
        
        Landmark groups:
        - Jaw: 0-16
        - Right eyebrow: 17-21
        - Left eyebrow: 22-26
        - Nose: 27-35
        - Right eye: 36-41
        - Left eye: 42-47
        - Mouth outer: 48-59
        - Mouth inner: 60-67
        """
        if not self.use_landmarks:
            return None
        
        import cv2
        import numpy as np
        
        # Convert tensor to numpy
        img_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        faces = self.detector(gray)
        if len(faces) == 0:
            return None
        
        landmarks = self.predictor(gray, faces[0])
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Define regions based on landmarks
        regions = {
            'left_eye': points[36:42],
            'right_eye': points[42:48],
            'left_eyebrow': points[17:22],
            'right_eyebrow': points[22:27],
            'nose': points[27:36],
            'mouth': points[48:68],
            'jaw': points[0:17],
        }
        
        return regions
    
    def analyze_region_from_landmarks(self, heatmap, landmarks, region_points):
        """
        Analyze a specific region defined by landmark points.
        """
        import cv2
        import numpy as np
        
        H, W = heatmap.shape
        
        # Create mask for this region
        mask = np.zeros((H, W), dtype=np.uint8)
        hull = cv2.convexHull(region_points)
        cv2.fillConvexPoly(mask, hull, 1)
        
        # Apply mask to heatmap
        region_heatmap = heatmap * mask
        region_pixels = region_heatmap[mask > 0]
        
        if len(region_pixels) == 0:
            return 0.0, 0.0
        
        avg_score = region_pixels.mean()
        max_score = region_pixels.max()
        
        return avg_score, max_score
