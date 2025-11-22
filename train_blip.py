import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import datetime
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from dataloaders.faceswapDataset import FaceswapDataset
from dataloaders.faceswapImagesDatasetAligned import FaceswapImagesDatasetAligned
from dataloaders.faceswapImagesDataset import FaceswapImagesDataset
from models.alignment_pretrained.model_with_bce_images_blip import MMModerator
from util import seed_worker, set_seed, compute_eer
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve,roc_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import silhouette_score
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from models.DinoLORA import DINOEncoderLoRA
# torch.autograd.set_detect_anomaly(True)

from models.alignment_pretrained.unet import UNetImageDecoder
import torchvision
import random
from datetime import timedelta
from transformers import get_linear_schedule_with_warmup
import cv2
from PIL import Image
from models.CLIPEncoder import CLIPEncoder
import torchvision.transforms.functional as TF
from typing import Dict, List, Optional
def to_pil(tensor):
    # Handle [B,C,H,W] or [C,H,W]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)   # drop batch
    if tensor.dtype != torch.uint8:
        tensor = (tensor.clamp(0,1) * 255).to(torch.uint8)
    return TF.to_pil_image(tensor)

# torch.set_printoptions(threshold=10_0000)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def compute_caption_metrics(generated_captions, ground_truth_captions):
    """
    Compute metrics to evaluate caption quality.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge import Rouge
    
    rouge = Rouge()
    bleu_scores = []
    rouge_scores = []
    
    for gen, gt in zip(generated_captions, ground_truth_captions):
        # BLEU score
        reference = [gt.split()]
        candidate = gen.split()
        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        bleu_scores.append(bleu)
        
        # ROUGE score
        if len(gen) > 0 and len(gt) > 0:
            rouge_score = rouge.get_scores(gen, gt)[0]
            rouge_scores.append(rouge_score['rouge-l']['f'])
    
    return {
        'bleu': np.mean(bleu_scores),
        'rouge-l': np.mean(rouge_scores)
    }

def create_vision_encoder():
    REPO_DIR = "/dinov3"
    model = CLIPEncoder()
    return model
   


def unnormalize(t: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    t: [C, H, W] or [B, C, H, W], normalized to ImageNet stats.
    Returns: unnormalized tensor in [0,1] range (clamped).
    """
    mean = torch.tensor(mean, device=t.device).view(-1, 1, 1)
    std = torch.tensor(std, device=t.device).view(-1, 1, 1)
    return (t * std + mean).clamp(0, 1)

def _to_tchw(arr4d):
    """
    Accepts 4‑D array in any of the common orders and returns (T,C,H,W).
    """
    if arr4d.ndim != 4:
        raise ValueError("Need a 4‑D array/tensor")

    s = arr4d.shape
    # try to find the axis with 1 or 3 → that *must* be channels
    chan_axes = [ax for ax, dim in enumerate(s) if dim in (1, 3)]
    if not chan_axes:
        raise ValueError("Could not identify channel axis (no dim of size 1 or 3)")
    c_ax = chan_axes[0]

    # heuristically pick a *frame* axis: smallest dim that is NOT channel and ≤ 500
    frame_axes = [ax for ax in range(4) if ax != c_ax and s[ax] <= 500]
    if not frame_axes:
        frame_axes = [ax for ax in range(4) if ax != c_ax]   # fallback: just pick first
    t_ax = frame_axes[0]

    # remaining two axes are height / width
    rem_axes = [ax for ax in range(4) if ax not in (t_ax, c_ax)]
    h_ax, w_ax = rem_axes

    arr_tchw = np.transpose(arr4d, (t_ax, c_ax, h_ax, w_ax))
    return arr_tchw  # (T,C,H,W)

def _save_with_cv2(video_4d, path="recon.mp4", fps=25):
    """
    video_4d : torch.Tensor | np.ndarray  (any of the 4 common layouts)
               float∈[-1,1]∪[0,1]  or uint8
    """
    # ---- Torch → NumPy ----------------------------------------------------
    if isinstance(video_4d, torch.Tensor):
        arr = video_4d.detach().cpu().numpy()
    else:
        arr = video_4d

    # ---- Re‑order to (T,C,H,W) -------------------------------------------
    arr = _to_tchw(arr)                         # (T,C,H,W)
    T, C, H, W = arr.shape

    # ---- Float → uint8 ----------------------------------------------------
    if issubclass(arr.dtype.type, np.floating):
        if arr.min() < 0:
            arr = (arr + 1.0) / 2.0             # [-1,1] → [0,1]
        arr = (arr.clip(0, 1) * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # ---- (T,C,H,W) → (T,H,W,C)  for OpenCV -------------------------------
    arr = np.transpose(arr, (0, 2, 3, 1))       # (T,H,W,C)

    # ---- Write with OpenCV -----------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))

    for frame in arr:
        if C == 1:                              # gray → 3‑channel
            frame = np.repeat(frame, 3, axis=2)
        frame_bgr = np.ascontiguousarray(frame[:, :, ::-1])  # RGB→BGR, positive strides
        writer.write(frame_bgr)

    writer.release()
    print(f"✅ Saved reconstructed clip ➜ {path}")


def save_reconstructed_video(tensor_4d, path="recon.mp4", fps=25):
    """
    tensor_4d: (T, C, H, W) or (C, T, H, W) — float32 or float32 in [0,1] or [-1,1].
    Saves to .mp4 using torchvision.
    """
    import torchvision

    vid = tensor_4d.detach().cpu()

    # Convert (C, T, H, W) → (T, C, H, W)
    if vid.shape[0] in (1, 3):
        vid = vid.permute(1, 0, 2, 3)  # now (T, C, H, W)

    if vid.dtype in (torch.float32, torch.float32):
        if vid.min() < 0:
            vid = (vid + 1) / 2  # convert from [-1, 1] to [0, 1]
        vid = (vid.clamp(0, 1) * 255).to(torch.uint8)  # to [0, 255]

    # (T, C, H, W) → (T, H, W, C)
    vid = vid.permute(0, 2, 3, 1).contiguous()  # must be contiguous

    # Fix: ensure it's a torch.Tensor on CPU with uint8 dtype
    if not isinstance(vid, torch.Tensor):
        vid = torch.from_numpy(vid)
    if vid.device.type != "cpu":
        vid = vid.cpu()
    if vid.dtype != torch.uint8:
        vid = vid.to(torch.uint8)

    torchvision.io.write_video(filename=path, video_array=vid, fps=fps, video_codec='libx264')
    print(f"✅ Saved reconstructed video ➜ {path}")


class CL(torch.nn.Module):
    def __init__(self, config, bit):
        super(CL, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.bit = bit

    def forward(self, h1, h2, weighted, labels):
        try:
            logits = torch.einsum('ik,jk->ij', h1, h2)
            logits = logits / self.bit / 0.3

            balance_logits = h1.sum(0) / h1.size(0)
            reg = self.mse(balance_logits, torch.zeros_like(balance_logits)) - self.mse(h1, torch.zeros_like(h1))
            weighted = torch.where(
                labels == 0,
                torch.zeros_like(weighted),
                weighted
            )
            loss = self.ce(logits, weighted.long()) + reg
        except Exception as e:
            print(f"Contrastive loss error: {e}")
            loss = torch.tensor(1.0, requires_grad=True).to(h1.device)
        return loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, use_logits=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_logits = use_logits

    def forward(self, logits, targets):
        if self.use_logits:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def smooth_labels(labels, smoothing=0.1):
    assert 0 <= smoothing < 1
    with torch.no_grad():
        labels = labels * (1 - smoothing) + smoothing / 2
    return labels


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
import random

def random_drop(batch, drop_prob=0.2, keys=("mfcc", "video")):
    """
    For every key in `keys` that is not None, drop it with `drop_prob`
    BUT never drop all of them in the same call.
    """
    present = [k for k in keys if batch[k] is not None]          # modalities that actually exist
    if len(present) <= 1:                                        # nothing to drop or only one present
        return batch

    # decide independently which present modalities to drop
    to_drop = [k for k in present if random.random() < drop_prob]

    # make sure at least one modality stays
    if len(to_drop) == len(present):
        keep_one = random.choice(to_drop)
        to_drop.remove(keep_one)

    # apply the drops
    for k in to_drop:
        batch[k] = None
        if f"{k}_aug" in batch:          # also clear *_aug if it exists
            batch[f"{k}_aug"] = None

    return batch

class UncertaintyWeighting(nn.Module):
    """
    Learns task-dependent uncertainty parameters that automatically balance losses.
    Based on "Multi-Task Learning Using Uncertainty to Weight Losses" (Kendall et al.)
    """
    def __init__(self, num_tasks: int):
        super().__init__()
        # Log-variance parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        losses: List of loss tensors for each task
        Returns: Weighted sum of losses + regularization term
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss / len(losses)


def main(args):
    
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30) )  # Initialize DDP

    local_rank = dist.get_rank() #int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # device = torch.device(f"cuda")
    # torch.cuda.set_device(local_rank)
    
    torch.backends.cudnn.benchmark = True  # selects best conv algos for your input sizes
    torch.backends.cudnn.enabled   = True
    scaler = torch.cuda.amp.GradScaler()
    resume_training = args.load_from is not None

    # Create the dataset
    path = args.base_dir






    def collate_fn(batch):
        # 1) turn each sample tuple into a unified dict (as before)…
        unified = []
        for sample in batch:
            if len(sample) == 12:
                (mfcc, mfcc_aug, audio, video, video_aug,
                text, landmarks, label, filenames, _, multi_label, flow) = sample
                d = {
                    "mfcc":      mfcc,      "mfcc_aug": mfcc_aug,
                    "audio":     None,     "video":     video,
                    "video_aug": video_aug, "text":      None,
                    "landmarks": None, "flow":      None,
                    "images":    None,      "images_aug":None,
                    "labels":     label,"multi_label": multi_label
                }
            elif len(sample) == 6:
                video,video_aug,fake_video, fake_video_aug, label, diff_video = sample
                
                d = {
                    "mfcc":      None,      "mfcc_aug": None,
                    "audio":     None,      "video":     video,
                    "video_aug": fake_video,      "text":      None,
                    "landmarks": None,      "flow":      None,
                    "images":    None,    "images_aug":None,
                    "labels":     label,"multi_label": diff_video
                }
                
            elif len(sample) == 5:
                real_image, fake_image, diff_image,pixels_rgb_t, label = sample
                d = {
                    "mfcc":      None,      "mfcc_aug": None,
                    "audio":     None,      "video":     None,
                    "video_aug": None,      "text":      None,
                    "landmarks": None,      "flow":      diff_image,
                    "images":    real_image,    "images_aug":fake_image,
                    "labels":     label,"multi_label":diff_image
                }
            elif len(sample) == 3:
                image, image_aug, label = sample
                d = {
                    "mfcc":      None,      "mfcc_aug": None,
                    "audio":     None,      "video":     None,
                    "video_aug": None,      "text":      None,
                    "landmarks": None,      "flow":      None,
                    "images":    image,    "images_aug":image_aug,
                    "labels":     label, "multi_label":None
                }
                # print("these are 4")
                # exit()
            
            else:
                raise ValueError(f"Unexpected sample length {len(sample)}")
            unified.append(d)

        # 2) batch each field only if it's a Tensor; leave others as lists
        batched = {}
        for k in unified[0].keys():
            vals = [d[k] for d in unified]
            # if nobody in the batch has that modality
            if all(v is None for v in vals):
                batched[k] = None
                continue

            # find first non-None entry
            template = next(v for v in vals if v is not None)

            if isinstance(template, torch.Tensor):
                # fill any Nones with zero-tensors of the right shape
                filled = []
                for v in vals:
                    if v is None:
                        filled.append(torch.zeros_like(template))
                    else:
                        filled.append(v)
                # print(filled)
                batched[k] = torch.stack(filled, dim=0)
            else:
                # non-Tensor field: keep the raw list (ints, strings, etc.)
                batched[k] = vals

        return batched




    def make_val_loader(dataset, args):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,   # for validation usually keep all samples
            collate_fn=collate_fn
        )
        return loader

    faceswap_val_video_dataset = FaceswapImagesDataset(path, partition="test", take_datasets=["NeuralTextures"])
    faceswap_video_df_val = FaceswapImagesDataset(path, partition="test", take_datasets=["Deepfakes"])
    faceswap_video_f2f_val = FaceswapImagesDataset(path, partition="test", take_datasets=["Face2Face"])
    faceswap_video_fs_val = FaceswapImagesDataset(path, partition="test", take_datasets=["FaceShifter"])
    faceswap_video_fw_val = FaceswapImagesDataset(path, partition="test", take_datasets=["FaceSwap"])

    faceswap_video_dfd_val = FaceswapImagesDataset(path, partition="test", take_datasets=["DFD_simple"])
    faceswap_video_wf_val = FaceswapImagesDataset(path, partition="test", take_datasets=["WildFake"])
    faceswap_video_DFDCp_val = FaceswapImagesDataset(path, partition="test", take_datasets=["DFDCp"])
    faceswap_video_DF0_val = FaceswapImagesDataset(path, partition="test", take_datasets=["DF0"])
    faceswap_video_FFIW_val = FaceswapImagesDataset(path, partition="test", take_datasets=["FFIW"])
    faceswap_video_simswap_val = FaceswapImagesDataset(path, partition="test", take_datasets=["simswap"])
    faceswap_video_blendface_val = FaceswapImagesDataset(path, partition="test", take_datasets=["blendface"])
    faceswap_video_34s_val = FaceswapImagesDataset(path, partition="test", take_datasets=["e4s"])
    faceswap_video_facedancer_val = FaceswapImagesDataset(path, partition="test", take_datasets=["facedancer"])
    faceswap_video_fsgan_val = FaceswapImagesDataset(path, partition="test", take_datasets=["fsgan"])
    faceswap_video_inswap_val = FaceswapImagesDataset(path, partition="test", take_datasets=["inswap"])
    faceswap_video_mobileswap_val = FaceswapImagesDataset(path, partition="test", take_datasets=["inswap"])
    faceswap_video_uniface_val = FaceswapImagesDataset(path, partition="test", take_datasets=["uniface"])


    
    
    faceswap_video_dfdc_val = FaceswapImagesDataset(path, partition="test", take_datasets=["DFDC"])
    faceswap_video_dfdc_test_val = FaceswapImagesDataset(path, partition="test", take_datasets=["DFDC_test"])
    faceswap_video_celebdfv2_test_val = FaceswapImagesDataset(path, partition="test", take_datasets=["CelebDFV2"])

    faceswap_video_simswap_val_loader = make_val_loader(faceswap_video_simswap_val, args)
    faceswap_video_blendface_val_loader = make_val_loader(faceswap_video_blendface_val, args)
    faceswap_video_34s_val_loader = make_val_loader(faceswap_video_34s_val, args)
    faceswap_video_facedancer_val_loader = make_val_loader(faceswap_video_facedancer_val, args)
    faceswap_video_fsgan_val_loader = make_val_loader(faceswap_video_fsgan_val, args)
    faceswap_video_inswap_val_loader = make_val_loader(faceswap_video_inswap_val, args)
    faceswap_video_mobileswap_val_loader = make_val_loader(faceswap_video_mobileswap_val, args)
    faceswap_video_uniface_val_loader = make_val_loader(faceswap_video_uniface_val, args)
    
    faceswap_video_val_loader = make_val_loader(faceswap_val_video_dataset, args)
   
                        
    faceswap_video_df_val_loader = make_val_loader(faceswap_video_df_val, args)


    faceswap_video_f2f_val_loader = make_val_loader(faceswap_video_f2f_val, args)

    faceswap_video_fs_val_loader = make_val_loader(faceswap_video_fs_val, args)

    faceswap_video_fw_val_loader = make_val_loader(faceswap_video_fw_val, args)

               
    faceswap_video_dfd_val_loader = make_val_loader(faceswap_video_dfd_val, args)


    faceswap_video_wf_val_loader = make_val_loader(faceswap_video_wf_val, args)

    
    faceswap_video_DFDCp_val_loader = make_val_loader(faceswap_video_DFDCp_val, args)


    faceswap_video_DF0_val_loader = make_val_loader(faceswap_video_DF0_val, args)


    faceswap_video_FFIW_val_loader = make_val_loader(faceswap_video_FFIW_val, args)

    

    faceswap_video_dfdc_val_loader = make_val_loader(faceswap_video_dfdc_val, args)
    faceswap_video_dfdc_test_val_loader = make_val_loader(faceswap_video_dfdc_test_val, args)
    faceswap_video_celebdfv2_test_val_loader = make_val_loader(faceswap_video_celebdfv2_test_val, args)
    
    faceswap_video_train_dataset = FaceswapImagesDatasetAligned(path, partition="train", take_datasets=["NeuralTextures","Deepfakes", "Face2Face","FaceSwap"])
    # exit()

# ,train_dataset,faceswap_video_train_dataset
    combined_ds = ConcatDataset([faceswap_video_train_dataset])
    # combined_ds = ConcatDataset([train_dataset, faceswap_video_train_dataset, faceswap_video_df, faceswap_video_fw]) #GPU 4

    # combined_ds = ConcatDataset([faceswap_video_train_dataset])
    train_sampler = torch.utils.data.distributed.DistributedSampler(combined_ds)
    train_loader = DataLoader(combined_ds, batch_size=args.batch_size,
                          sampler=train_sampler,
                          num_workers=args.num_workers,
                          pin_memory=True,
                          drop_last=True, 
                          collate_fn=collate_fn
                          )

    # exit()


    pretraining = False
    num_classes = 1
    vision_encoder = create_vision_encoder()
    vision_encoder.to(device=device, dtype=torch.float32)
    unet_decoder = UNetImageDecoder(
            num_patches=256,       # 7 × 7 grid (ViT-B/32)
            token_dim=1024,        # ViT-B/32 embedding dim
            out_channels=3,       # mask or 3 for RGB
            base_channels=256,
            img_size=256,
            grid_hw=(16, 16)        # explicitly set to match patch grid
        )
        
    unet_decoder.to(device=device, dtype=torch.float32)
        
    model = MMModerator(pretraining=pretraining,vision_encoder=vision_encoder, unet_decoder=unet_decoder, num_classes=num_classes)
    model = model.to(device=device, dtype=torch.float32)
    total_parameters = count_parameters(model=model)
    print(f"Model created. Trainable parameters: {total_parameters / 1e6:.2f}M")

    base_lr = 3e-4  # Fixed learning rate for AdamW


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,             # 0.00005 (safe for partial fine-tuning)
        betas=(0.9, 0.999),  # standard AdamW (good for ViTs)
        weight_decay=0.05,   # higher than 1e-4; improves generalization for ViTs
        eps=1e-8             # keep default (numerical stability)
    )
  
    # ----------------  Scheduler Setup  ----------------
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(2000, total_steps // 10)  # 10% of total steps or 2000, whichever is smaller

    # Option 1: Linear warmup + Cosine decay (Recommended)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )



    

    total_steps  = args.epochs * len(train_loader)
    # decay LR by 0.1 every 30 epochs
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)


    start_epoch = 0
    def reset_weights(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        

    if resume_training:
        checkpoint_path = os.path.join(args.load_from, "checkpoints", "model_state.pt")
        checkpoint_path_encoder = os.path.join(args.load_from, "checkpoints", "model_state_encoder.pt")
        checkpoint_path_decoder = os.path.join(args.load_from, "checkpoints", "model_state_decoder.pt")
        raw   = torch.load(checkpoint_path, map_location=device)
        raw_encoder   = torch.load(checkpoint_path_encoder, map_location=device)
        raw_decoder   = torch.load(checkpoint_path_decoder, map_location=device)
        # model.load_state_dict(state, strict=True)
        # if you saved the full dict with optimizer & friends, pull out the model_state
        
        
        sd    = raw.get("model_state_dict", raw)

        # strip off any "module." prefixes
        new_sd = {}
        for k, v in sd.items():
            new_key = k.replace("module.", "")  
            new_sd[new_key] = v
            
        model.load_state_dict(new_sd)        # strict=True by default
        

        
        sd_encoder    = raw_encoder.get("model_state_dict", raw_encoder)

        # strip off any "module." prefixes
        new_sd_encoder = {}
        for k, v in sd_encoder.items():
            new_key = k.replace("module.", "")  
            new_sd_encoder[new_key] = v
            
        vision_encoder.load_state_dict(new_sd_encoder)        # strict=True by default
        
        sd_decoder    = raw_decoder.get("model_state_dict", raw_decoder)

        # strip off any "module." prefixes
        new_sd_decoder = {}
        for k, v in sd_decoder.items():
            new_key = k.replace("module.", "")  
            new_sd_decoder[new_key] = v
            
        unet_decoder.load_state_dict(new_sd_decoder)        # strict=True by default
        
        # model.load_state_dict(state, strict=False)
        # optimizer.load_state_dict(raw['optimizer_state_dict'])
        # scheduler.load_state_dict(raw['scheduler_state_dict'])
        

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    

    log_dir = os.path.join(args.log_dir, args.encoder)
    os.makedirs(log_dir, exist_ok=True)
   
    
    if dist.get_rank() == 0:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(log_dir, current_time)
        os.makedirs(log_dir, exist_ok=True)

    # writer = SummaryWriter(log_dir=log_dir)
    if dist.get_rank() == 0:
        writer   = SummaryWriter(log_dir=log_dir)
        # model.apply(reset_weights)
    else: 
        writer = None
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        f.write(str(vars(args)))

    bit = 64
    config = None
    criterion = nn.BCEWithLogitsLoss()
    criterion_contrastive = CL(config, bit)
    CLASSIFICATION_WEIGHT = 1
    CONTRASTIVE_WEIGHT = 1
    best_val_eer = 1.0
    global_step = 0
    l2_lambda = 0.001



    intra_w_start     = 0.0
    intra_w_end       = 0.0
    inter_w_start     = 0.0
    inter_w_end       = 0.0
    lipsync_w_start   = 0.0
    lipsync_w_end     = 0.0
    moe_start   = 0.0
    moe_end     = 0.0
    
    llm_w_start     = 0.6
    llm_w_end       = 0.6
    recon_w_start     = 0.4
    recon_w_end       = 0.4
    semantic_w_start = 0.0
    semantic_w_end = 0.0
    cls_w_start       = 0.9
    cls_w_end         = 0.9  # keep classification weight fixed
    #GPU 4
    balancer = UncertaintyWeighting(num_tasks=4)
    # Initialize BERT tokenizer (using "bert-base-uncased")

    best_threshold=0.5

    for epoch in range(start_epoch, args.epochs):
        model.train()
        model.total_samples =  len(train_loader)
        # model.total_samples = model.total_samples + len(faceswap_train_loader)
        train_loss_epoch = 0.0
        correct_predictions = 0
        total_samples = 0
        pos_samples, neg_samples, all_preds, all_labels = [], [], [], []
        grad_norms = []
        train_sampler.set_epoch(epoch)
        model.pretraining = pretraining
        for loader_info in [{"loader":train_loader,"name":"VideoAudio"}]:#{"loader": faceswap_train_loader, "name": "faceswap"},,]:,{"loader":faceswap_video_train_loader,"name":"faceswapVideo"}
            # break
            data_loader = loader_info['loader']
            # print(data_loader)
            loader_name = loader_info['name']
            
            for i, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch + 1}/{args.epochs} {loader_name}")):
                
                # if args.debug and i > 500:
                #     break
                mfcc,mfcc_aug, audio, video,video_aug, text_tokens, landmarks, label, filenames, _ ,multi_label, optical_flow= batch
                # batch = random_drop(batch, drop_prob=0.2, keys=("mfcc", "video"))
                for mod in batch:
                    if batch[mod] is not None: 
                        if mod=="labels":
                            label = batch[mod] = batch[mod].to(device=device, dtype=torch.float32)
                        else:
                            batch[mod]=batch[mod].to(device=device, dtype=torch.float32)
                

                optimizer.zero_grad()
                # with autocast("cuda", dtype=torch.bfloat32):
                # with torch.amp.autocast(device_type=“cuda”, dtype=torch.float32):
                logits, losses, label, image_recon, gt_captions, captions, captions_readable, overlay, overlay_ori = model(**batch)
           
                progress = global_step / (total_steps - 1)    # in [0,1]
                        
                w_intra   = intra_w_start   + progress * (intra_w_end   - intra_w_start)
                w_inter   = inter_w_start   + progress * (inter_w_end   - inter_w_start)
                w_lipsync = lipsync_w_start + progress * (lipsync_w_end - lipsync_w_start)
                w_moe = moe_start + progress * (moe_end - moe_start)
                w_recon   = recon_w_start   + progress * (recon_w_end   - recon_w_start)
                w_llm   = llm_w_start   + progress * (llm_w_end   - llm_w_start)
                w_semantic   = semantic_w_start   + progress * (semantic_w_end   - semantic_w_start)
                
                w_cls     = cls_w_start     + progress * (cls_w_end     - cls_w_start)
                

                intra_sum = sum(losses["intra"].values())
                inter_sum = sum(losses["inter"].values())
                lipsyncLoss = sum(losses["lipsyncLoss"].values())
                reconstructionLoss = sum(losses["reconstruction"].values())
                llm_loss = sum(losses["llm_loss"].values())
                llm_samantic_loss = sum(losses["llm_samantic_loss"].values())
                cls_loss = sum(losses["cls_loss"].values())
            

                total_loss = (
                    w_intra   * intra_sum
                    + w_inter   * inter_sum
                    + w_lipsync * lipsyncLoss
                    + w_recon   * reconstructionLoss
                    + w_llm * llm_loss
                    + w_semantic * llm_samantic_loss 
                    # + w_moe * moeLoss
                    )
                # print("cls_loss",cls_loss)
                loss = w_cls * cls_loss + total_loss * (1-w_cls)
            


                #  Mixed precision backprop
                scaler.scale(loss).backward()

                scaler.step(optimizer)   # this will unscale internally if not done already
                scaler.update()
                scheduler.step()

                global_step += 1

                total_norm = 0
                
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                grad_norms.append(total_norm)
            
                pred = torch.sigmoid(logits)
                
                if writer is not None:
                    print(pred.min(), pred.max(), pred.mean())
                    print("Labels are: ",torch.unique(label, return_counts=True))
                    if total_norm < 1e-5 or total_norm > 1e3:
                        print("Unstable gradient norm:", total_norm)
                    print("Logits:", logits[:5])
                    print("Pred min and max",pred.min(), pred.max(), pred.mean())
                    print("Preds:", torch.sigmoid(logits)[:5])
                    print("Labels:", label[:5])

                # print(pred.device, label.device)
                pos_samples.append(pred[label == 1].detach().cpu().numpy())
                neg_samples.append(pred[label == 0].detach().cpu().numpy())
                pred_class = (pred > best_threshold).long().squeeze()
                correct_predictions += (pred_class == label).sum().item()
            
                all_preds.extend(pred.detach().cpu().numpy())
                all_labels.extend(label.detach().cpu().numpy())
                
                total_samples += label.size(0)
                train_loss_epoch += loss.item()

                if writer is not None:
                    
                    # if i==10:
                    #     break
                    if i < 4:
                        
                        # import csv
                        save_dir = os.path.join(log_dir,  str(epoch), str(i))
                        os.makedirs(save_dir, exist_ok=True) 

                        # Specify the output TXT file name
                        output_file = f"{log_dir}/{epoch}/{i}/{loader_name}_prompts.txt"

                        # Open the file in write mode
                        with open(output_file, 'w', encoding='utf-8') as f:
                            for line in captions:
                                if isinstance(line, (list, tuple)):
                                    # If caption is nested (e.g. [['text1'], ['text2']])
                                    f.write(' '.join(map(str, line)) + '\n')
                                else:
                                    f.write(str(line) + '\n')

                        # Specify the output TXT file name
                        output_file = f"{log_dir}/{epoch}/{i}/{loader_name}_gt_prompts.txt"

                        # Open the file in write mode
                        with open(output_file, 'w', encoding='utf-8') as f:
                            for line in gt_captions:
                                if isinstance(line, (list, tuple)):
                                    # If caption is nested (e.g. [['text1'], ['text2']])
                                    f.write(' '.join(map(str, line)) + '\n')
                                else:
                                    f.write(str(line) + '\n')
                                    
                        print(f"List successfully written to {output_file}")
                     
                        # Specify the output TXT file name
                        output_file = f"{log_dir}/{epoch}/{i}/{loader_name}_captions_readable_prompts.txt"

                        # Open the file in write mode
                        with open(output_file, 'w', encoding='utf-8') as f:
                            for line in captions_readable:
                                if isinstance(line, (list, tuple)):
                                    # If caption is nested (e.g. [['text1'], ['text2']])
                                    f.write(' '.join(map(str, line)) + '\n')
                                else:
                                    f.write(str(line) + '\n')
                                    
                        print(f"List successfully written to {output_file}")
                     


                        

                        
                        for index, v in enumerate(overlay[len(overlay)//2:]):
                            img = v.clamp(0, 1)   # mask, already [0,1]
                            # img = unnormalize(img)
                            to_pil(img).save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_overlay.png")

                        
                        for index, v in enumerate(overlay_ori[len(overlay_ori)//2:]):
                            img = v.clamp(0, 1)   # mask, already [0,1]
                            # img = unnormalize(img)
                            to_pil(img).save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_overlay_ori.png")

                        for index, v in enumerate(batch["multi_label"]):
                            img = v.clamp(0, 1)   # mask, already [0,1]
                            # img = unnormalize(img)
                            to_pil(img).save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_pixel_rgb.png")

                            
                        for index, v in enumerate(batch["flow"]):
                            img = v.clamp(0, 1)   # mask, already [0,1]
                            # img = unnormalize(img)
                            to_pil(img).save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_pixel_diff.png")

                            

                        for index, image in enumerate(batch["images"]):
                            img = unnormalize(image)   # undo normalization
                            to_pil(img).save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_ori.png")

                        for index, image in enumerate(batch["images_aug"]):
                            img = unnormalize(image)
                            to_pil(img).save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_fake.png")

                        for index, v in enumerate(image_recon[len(image_recon)//2:]):
                          
                            image_pil = to_pil(img)
                            image_pil.save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_recon.png")
                            
                            img = v.clamp(0,1)
                            # img = v.sigmoid().clamp(0,1)
                            image_pil = to_pil(img)
                            image_pil.save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_recon_clamp.png")

                            
                            # img = v.clamp(0,1)
                            img = v.sigmoid().clamp(0,1)
                            image_pil = to_pil(img)
                            image_pil.save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_recon_sigmoid.png")

                            
                            image_pil = to_pil(v / v.max())
                            image_pil.save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_recon_v_max.png")
                                
            
                            # img = v.clamp(0,1)
                            # img = v.sigmoid().clamp(0,1)
                            img = unnormalize(v)
                            image_pil = to_pil(img)
                            image_pil.save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_recon_unormalized.png")
                    
           
                        # break

                                
                    # epoch * len(data_loader) + i
                    # epoch * len(data_loader) + i
                    writer.add_scalar("Loss/train", loss.item(),global_step)
                    writer.add_scalar("Loss/recon", reconstructionLoss.item(),global_step)
                    writer.add_scalar("Loss/clip", llm_loss,global_step)
                    writer.add_scalar("GradNorm/train", total_norm, global_step)
                    writer.add_scalar("LR/train", optimizer.param_groups[0]['lr'], global_step)

        avg_train_loss = train_loss_epoch / total_samples
        
        train_accuracy = correct_predictions / total_samples
    
        if not pretraining:
            roc_auc = roc_auc_score(all_labels, all_preds)
            precision, recall, _ = precision_recall_curve(all_labels, all_preds)
            auc_pr = auc(recall, precision)
            # fpr, tpr, thresholds = roc_curve(all_labels, all_preds, pos_label=2)
            # auc_pr = auc(fpr, tpr)

            ap_score = average_precision_score(all_labels, all_preds)
            train_eer = compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]

        if writer is not None:
            if not pretraining:
                # writer.add_scalar("Train/AvgLoss", avg_train_loss, epoch)
                writer.add_scalar("Train/Accuracy", train_accuracy, epoch)
                writer.add_scalar("Train/ROC_AUC", roc_auc, epoch)
                writer.add_scalar("Train/AUC_PR", auc_pr, epoch)
                writer.add_scalar("Train/AP", ap_score, epoch)
                writer.add_scalar("EER/train", train_eer, epoch)

        if writer is not None:
            writer.add_scalar("Train/AvgLoss", avg_train_loss, epoch)
        # Save a histogram of model weights (for every 5 epochs)
        if epoch % 5 == 0:
            for name, param in model.named_parameters():
                if writer is not None:
                    writer.add_histogram(name, param.data.cpu().numpy(), epoch)

        if dist.get_rank() == 0:
            # Save training state
            state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }
            torch.save(state, os.path.join(checkpoint_dir, f"model_state.pt"))
            
            state = {
                    'epoch': epoch,
                    'model_state_dict': vision_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }
            torch.save(state, os.path.join(checkpoint_dir, f"model_state_encoder.pt"))


            state = {
                    'model_state_dict': unet_decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }
            torch.save(state, os.path.join(checkpoint_dir, f"model_state_decoder.pt"))



    # Evaluate on the validation set.
        model.eval()
        # {"loader":faceswap_video_val_loader,"name":"faceswapVideo"}
        for loader_info in [
                            # {"loader":dev_loader,"name":"VideoAudio"},
                            # {"loader": dev_loader_v2, "name": "deepspeak_v2"},
                            {"loader": faceswap_video_simswap_val_loader, "name": "SimSwap"},
                            {"loader": faceswap_video_blendface_val_loader, "name": "BlendFace"},
                            # {"loader": faceswap_video_34s_val_loader, "name": "34S"},
                            {"loader": faceswap_video_facedancer_val_loader, "name": "FaceDancer"},
                            {"loader": faceswap_video_fsgan_val_loader, "name": "FSGAN"},
                            {"loader": faceswap_video_inswap_val_loader, "name": "InSwap"},
                            {"loader": faceswap_video_mobileswap_val_loader, "name": "MobileSwap"},
                            {"loader": faceswap_video_uniface_val_loader, "name": "UniFace"},
                            # {"loader": faceswap_video_DF0_val_loader, "name": "DF0"},
                            {"loader": faceswap_video_FFIW_val_loader, "name": "FFIW"},
                            {"loader": faceswap_video_val_loader, "name": "NeuralTextures"},
                            # {"loader": dev_loader_timit, "name": "Deepfake_TIMIT"}
                            {"loader": faceswap_video_df_val_loader, "name": "Deepfakes"},
                            # {"loader": faceswap_video_f2f_val_loader, "name": "Face2Face"},
                            # {"loader": faceswap_video_fs_val_loader, "name": "FaceShifter"},
                            {"loader": faceswap_video_fw_val_loader, "name": "FaceSwap"},
                            {"loader": faceswap_video_dfd_val_loader, "name": "CelebDF"},
                            {"loader": faceswap_video_celebdfv2_test_val_loader, "name": "CelebDFV2"},
                            {"loader": faceswap_video_wf_val_loader, "name": "WildFake"},
                            {"loader": faceswap_video_DFDCp_val_loader, "name": "DFDCp"},

                            
                            {"loader": faceswap_video_dfdc_val_loader, "name": "DFDC_val"},
                            {"loader": faceswap_video_dfdc_test_val_loader, "name": "DFDC_test"},

                            
                            
                        ]:#,{"loader": train_loader, "name": "Audio-Visual"},]:
        # break
            data_loader = loader_info['loader']
            # print(data_loader)
            loader_name = loader_info['name']

            val_loss = 0
            correct_predictions = 0
            total_samples = 0
            pos_samples, neg_samples, all_preds, all_labels = [], [], [], []
            # model.pretraining = pretraining
            model.is_training=False
            with torch.no_grad():
                for i, batch in enumerate(tqdm(data_loader, desc="Validation")):

                    mfcc,mfcc_aug, audio, video,video_aug, text_tokens, landmarks, label, filenames, _ ,multi_label, optical_flow= batch
                    # batch = random_drop(batch, drop_prob=0.2, keys=("mfcc", "video"))
                    for mod in batch:
                        if batch[mod] is not None: 
                            if mod=="labels":
                                label = batch[mod] = batch[mod].to(device=device, dtype=torch.float32)
                            else:
                                batch[mod]=batch[mod].to(device=device, dtype=torch.float32)

                    classification_logits,losses, labels, image_recon, gt_captions,  captions, captions_readable, overlay , overlay_ori  = model(**batch)
                    intra_sum = sum(losses["intra"].values())
                    inter_sum = sum(losses["inter"].values())
                    lipsyncLoss = sum(losses["lipsyncLoss"].values())
                    reconstructionLoss = sum(losses["reconstruction"].values())
                    cls_loss = sum(losses["cls_loss"].values())

                    total_loss = (
                    intra_sum
                    + inter_sum
                    + lipsyncLoss
                    +  reconstructionLoss
                    )

                    loss = 0.5 * cls_loss + total_loss * 0.5
                    

                    val_loss += loss
                    pred = torch.sigmoid(classification_logits)
                    pred_class = (pred > best_threshold).long().squeeze()
                    correct_predictions += (pred_class == label).sum().item()
                    total_samples += label.size(0)
                    pos_samples.append(pred[label == 1].detach().cpu().numpy())
                    neg_samples.append(pred[label == 0].detach().cpu().numpy())
                    all_preds.extend(pred.detach().cpu().numpy())
                    all_labels.extend(label.detach().cpu().numpy())
                        
                    if writer is not None:
                        if i<4:
                                
                            save_dir = os.path.join(log_dir, str(epoch), str(i))
                            os.makedirs(save_dir, exist_ok=True) 
                                
                            # Specify the output TXT file name
                            output_file = f"{log_dir}/{epoch}/{i}/{loader_name}_prompts.txt"

                            # Open the file in write mode
                            with open(output_file, 'w', encoding='utf-8') as f:
                                for line in captions:
                                    if isinstance(line, (list, tuple)):
                                        # If caption is nested (e.g. [['text1'], ['text2']])
                                        f.write(' '.join(map(str, line)) + '\n')
                                    else:
                                        f.write(str(line) + '\n')

                            print(f"List successfully written to {output_file}")
                            

                            # Specify the output TXT file name
                            output_file = f"{log_dir}/{epoch}/{i}/{loader_name}_captions_readable_prompts.txt"

                            # Open the file in write mode
                            with open(output_file, 'w', encoding='utf-8') as f:
                                for line in captions_readable:
                                    if isinstance(line, (list, tuple)):
                                        # If caption is nested (e.g. [['text1'], ['text2']])
                                        f.write(' '.join(map(str, line)) + '\n')
                                    else:
                                        f.write(str(line) + '\n')
                                        
                            print(f"List successfully written to {output_file}")
                            
                            for index, v in enumerate(overlay):
                                img = v.clamp(0, 1)   # mask, already [0,1]
                                # img = unnormalize(img)
                                to_pil(img).save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_overlay.png")


                            for index,v in enumerate(image_recon):
                                images = unnormalize(batch["images"][index])
                                image_pil = to_pil(images)
                                image_pil.save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_ori.png")

                                
                                # img = unnormalize(image)   # undo normalization
                                
                                img = v.clamp(0,1)
                                # img = v.sigmoid().clamp(0,1)
                                image_pil = to_pil(img)
                                image_pil.save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_recon_clamp.png")

                                
                                # img = v.clamp(0,1)
                                img = v.sigmoid().clamp(0,1)
                                image_pil = to_pil(img)
                                image_pil.save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_recon_sigmoid.png")

                                
                                # img = v.clamp(0,1)
                                # img = v.sigmoid().clamp(0,1)
                                img = unnormalize(v)
                                image_pil = to_pil(img)
                                image_pil.save(f"{log_dir}/{epoch}/{i}/{loader_name}_{index}_recon_unormalized.png")
                                
                      
                

                val_loss /= len(data_loader)
                val_accuracy = correct_predictions / total_samples
         
                # print(all_labels)
                # scheduler.step(val_accuracy)
                if not pretraining:
                    val_eer = compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]
                    roc_auc_val = roc_auc_score(all_labels, all_preds)
                    precision_val, recall_val, _ = precision_recall_curve(all_labels, all_preds)
                    auc_pr_val = auc(recall_val, precision_val)
                    
                ap_score_val = average_precision_score(all_labels, all_preds)
             
                if writer is not None:
                    writer.add_scalar(f"{loader_name}/Val/AvgLoss", val_loss, epoch)
                    if not pretraining:
                        writer.add_scalar(f"{loader_name}/Val/Accuracy", val_accuracy, epoch)
                        writer.add_scalar(f"{loader_name}/Val/ROC_AUC", roc_auc_val, epoch)
                        writer.add_scalar(f"{loader_name}/Val/AUC_PR", auc_pr_val, epoch)
                        # writer.add_scalar("Val/AUC_R", auc_r_val, epoch)
                        writer.add_scalar(f"{loader_name}/Val/AP", ap_score_val, epoch)
                        writer.add_scalar(f"{loader_name}/EER/val", val_eer, epoch)
      


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="The number of epochs to train.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--gpu", type=int, default=0, help="The GPU to use.")
    parser.add_argument("--encoder", type=str, default="rawnet", help="The encoder to use.")
    parser.add_argument("--batch_size", type=int, default=24, help="The batch size for training.")
    parser.add_argument("--num_workers", type=int, default=6, help="The number of workers for the data loader.")
    parser.add_argument("--log_dir", type=str, default="logs", help="The directory for the logs.")
    parser.add_argument("--load_from", type=str, default=None, help="The path to the checkpoint to load from.")

    args = parser.parse_args()
    main(args)
