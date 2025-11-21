from __future__ import annotations


from typing import Dict, List, Optional, Sequence
from dataclasses import dataclass, field
from torch.cuda.amp import autocast
import transformers
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

from ..xbm.xbm_llava.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from ..xbm.xbm_llava.llava.model.language_model.llava_mpt import LlavaMptForCausalLM
from ..xbm.xbm_llava.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
)
from ..xbm.xbm_llava.llava.conversation import conv_templates, SeparatorStyle
from ..xbm.xbm_llava.llava.mm_utils import tokenizer_image_token


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def _set_all_frozen(model):
    for p in model.parameters():
        p.requires_grad = False

def _unfreeze_language_qkv_only(model):
    """
    Unfreezes only Q/K/V linear layers in the LANGUAGE transformer.
    Covers common naming across LLaMA/LLaVA, MPT, BLOOM, etc.
    """
    qkv_keys = (
        "model.layers.",             # anchor to language model in HF LLaMA/MPT
    )
    allow_tokens = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",   # LLaMA-style
        "attn.q_proj", "attn.k_proj", "attn.v_proj",                  # some variants
        "attn.Wqkv", "Wqkv",                                          # MPT-style fused QKV
        "query_key_value",                                            # BLOOM/others fused
    )

    for n, p in model.named_parameters():
        # keep everything frozen unless it’s clearly inside the language stack
        if not any(n.startswith(prefix) for prefix in qkv_keys):
            continue
        if any(tok in n for tok in allow_tokens):
            p.requires_grad = True
def _log_trainable(model, tag="model"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] trainable params: {trainable:,} / {total:,} "
          f"({100.0*trainable/total:.2f}%)")


def create_decoder(model_args, training_args, data_args):
    compute_dtype = torch.float32 if training_args.fp16 else (torch.bfloat32 if training_args.bf16 else torch.float32)

    if "mpt" in model_args.model_name_or_path:
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        model = LlavaMptForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
        )
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat32 if training_args.bf16 else None),
        )

    # ---- Freeze EVERYTHING immediately (before mm_mlp logic) ----
    _set_all_frozen(model)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Tokenizer
    if "mpt" in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    tokenizer.pad_token = tokenizer.unk_token

    # Vision modules
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.float32, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    # Config plumbing
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    # ---- Unfreeze ONLY Q/K/V in the *language* transformer ----
    _unfreeze_language_qkv_only(model)

    # ---- (Optional) also tune the mm_projector if requested ----
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        # keep everything else frozen, just open mm_projector too
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False  # re-freeze projector if asked

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # Debug print
    _log_trainable(model, tag="LLaVA")

    return model, tokenizer

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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14")
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    num_classes: int = field(default=1000)
    max_decode_length: int = field(default=50)
    min_decode_length: int = field(default=20)
    fixed_caps: bool = field(default=False)
    base_dir: Optional[str] = field(default="")
    batch_size: Optional[int] = field(default=16)
    log_dir: Optional[str] = field(default="")
    log_dir: Optional[str] = field(default="")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True, metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    fp16: bool = True
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    checkpoint: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default="result/temp")
    evaluate: Optional[bool] = field(default=False)
    max_epoch: Optional[int] = field(default=100)
    init_lr: Optional[float] = field(default=3e-5)
    min_lr: Optional[float] = field(default=0.0)
    lr_decay_rate: Optional[float] = field(default=0.9)
    warmup_step: Optional[int] = field(default=50)
    warmup_lr: Optional[float] = field(default=1e-6)
    cuda_device: Optional[str] = field(default="cuda")
    lambda_: Optional[float] = field(default=1.0)
    temperature: Optional[float] = field(default=10.0)
    temperature_annealing: Optional[str] = field(default="const")
    distributed: Optional[bool] = field(default=False)
    dist_url: Optional[str] = field(default="env://")
    world_size: Optional[int] = field(default=1)


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



@dataclass
class DataArguments:
    dataset: str = field(default="imagenet")
    image_size: int = field(default=336)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"


class MMModerator(nn.Module):
    def __init__(self, *, dim: int = 512, device: torch.device | str = "cpu",pretraining=False, vision_encoder=None, unet_decoder=None, num_classes=11):
        super().__init__()
        parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        self.device = torch.device(device)
        self.pretraining = pretraining
        dim=1024
        self.encoder_bottleneck = MaskEncoder(dim=dim, depth=12, heads=6, num_fusion_tokens=16,video_temporal_patch_size=4,video_channels=3,device=device) # Gpu 4 25.42 million parameters
        
        self.encoder_bottleneck.to(dtype=torch.float32)
        self.overlay_encoder = vision_encoder
        
        
        self.encoder = vision_encoder
        
        med_config="configs/bert_config.json",
        embed_dim=256
        clip_dim=1024
        prompt="Generate sentence describing details of visual objects in this image: "
        num_beams=3
        top_p=0.9
        repetition_penalty=1.0
        use_cross_attn=True
        momentum=1.0
        temperature=1

        #Text model initialization starts 
        med_config = "/media/NAS/USERS/shahid/MultimodalAudioVisualModerator/models/xbm/configs/bert_config.json"
        vision_width=1024

        # create the decoder
        self.text_decoder, self.tokenizer = create_decoder(model_args, training_args, data_args)
        self.text_decoder.config.use_cache = False

        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(self.text_decoder),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            self.text_decoder.to(torch.float32)
            print("Adding LoRA adapters...")
            self.text_decoder = get_peft_model(self.text_decoder, lora_config)

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = clip_dim
        self.text_encoder = BertModel.from_pretrained(
            "bert-large-uncased", config=encoder_config, add_pooling_layer=False
        )
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.text_encoder.to(torch.float32)
        text_width = self.text_encoder.config.hidden_size

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
        captions = self.generate_caption_from_heatmap(
            heatmap=heatmap, label=labels, threshold=0.45, diversity=0.65, n_variants=1, visual_embeds=visual_embeds, visual_mask=visual_mask, device=device
        )



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

        return decoder_input_ids, attention_mask, decoder_labels, captions#, semantic_emb

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
        return captions_seed
        # ----- BLIP refinement (batch) -----
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
        #         out = self.blip2_model.generate(**inputs)
        #         # inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float32)
        #         refined = self.blip2_processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        #         refined_captions.append(refined)
           

        #     return refined_captions
        # return captions_seed

        
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

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


    def preprocess(self, answers):
        prompts = ["<image>\n" + prompt for prompt in answers]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [
                tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                for prompt in prompts
            ],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        targets = input_ids.clone()
        instruction_len = len(tokenizer_image_token("<image>\n", self.tokenizer))
        for target in targets:
            target[:instruction_len] = IGNORE_INDEX
        return input_ids, targets
        
    def tokenize(self, batch_size):
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], self.prompt_base_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return input_ids.unsqueeze(0).repeat(batch_size, 1)

        
    @torch.no_grad()
    def infer(self, image, class_label=None, ret_caption=False, ret_attention=False):
        # Generate Caption
        input_ids = self.tokenize(batch_size=image.size(0)).to(image.device)
        # beam search
        decoder_output, image_embeds = self.text_decoder.generate(
            inputs=input_ids,
            images=image,
            image_sizes=[image.size],
            max_length=self.max_decode_length,
            min_length=self.min_decode_length,
            num_beams=self.num_beams,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=self.repetition_penalty,
            output_inputs_embeds=True,
        )
        captions = []
        for output in decoder_output:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        # Encode Caption (and Image Embedding)
        caption_text = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_decode_length,
            return_tensors="pt",
        ).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds.float(), "encoder_attention_mask": image_atts}
        cross_attn_kwargs = {"mode": "text"} if not self.use_cross_attn else model_kwargs
        caption_text_output = self.text_encoder(
            caption_text.input_ids,
            attention_mask=caption_text.attention_mask,
            return_dict=True,
            output_attentions=True,
            **cross_attn_kwargs,
        )
        text_feat = F.normalize(self.text_proj(caption_text_output.last_hidden_state[:, 0, :]), dim=-1)

        # Predict Final Labels
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
        decoder_input_ids, decoder_attention_mask, decoder_labels, generated_captions = \
            self.prepare_text_targets(overlay_ori, labels_expanded, max_length=self.max_decode_length, visual_embeds=embeddings_ori, visual_mask=embeddings_ori_m)
        
        with self.caption_context():
            decoder_input_ids, decoder_targets = self.preprocess(generated_captions)
            decoder_targets = decoder_targets.masked_fill(decoder_targets == self.tokenizer.pad_token_id, -100)
            decoder_outputs, image_embeds, embeded_targets = self.text_decoder(
                input_ids=decoder_input_ids.to(overlay.device),
                images=overlay,
                labels=decoder_targets.to(overlay.device),
                return_dict=True,
                output_inputs_embeds=True,
            )
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(overlay.device)
            tau = 0.7
            token_mask = nn.functional.gumbel_softmax(decoder_outputs.logits.float(), tau=tau, hard=True, dim=-1)
            indices = torch.tensor(range(0, token_mask.size(-1))).to(token_mask.device)
            decoded_tokens = (token_mask * indices).sum(-1)
            captions = []
            reduced_decoded_tokens = []
            for output, target in zip(decoded_tokens, embeded_targets):
                reduced_output = output[target != IGNORE_INDEX]
                caption = self.tokenizer.decode(reduced_output.detach(), skip_special_tokens=True)
                captions.append(caption)
                reduced_decoded_tokens.append(reduced_output)
            loss_lm = decoder_outputs.loss
            reduced_decoded_tokens = torch.nn.utils.rnn.pad_sequence(
                reduced_decoded_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        captions_readable = captions
        # Encode Caption (and Image Embedding)
        caption_text = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=reduced_decoded_tokens.size(1),
            return_tensors="pt",
        ).to(images.device)
     
        model_kwargs = {"encoder_hidden_states": embeddings_recon.to(dtype=torch.float32), "encoder_attention_mask": embeddings_m}
        cross_attn_kwargs = {"mode": "text"} if not self.use_cross_attn else model_kwargs
        caption_text_output = self.text_encoder(
            reduced_decoded_tokens,
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
            "moe_loss":{},
            "cls_loss":{}, 
            "kl_loss":{},
            "llm_samantic_loss":{}
        }

        
        losses["reconstruction"]["loss"] = mse_loss_images 
        student_emb = caption_text_output.last_hidden_state.mean(dim=1)

        if self.training:
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
