import numpy as np
import os
from torch.utils.data import Dataset
import librosa
import torch
import torchvision
import random
import cv2  # OpenCV for video processing
import torchvision.transforms as transforms
import traceback
import shutil
import torch.nn as nn
from pathlib import Path
from decord import VideoReader
from decord import cpu, gpu
import json
import torchvision.transforms as T
from skimage import transform as trans
import torch.nn.functional as F
import subprocess  # Missing import for extract_audio_with_ffmpeg
# from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Normalize, PolarityInversion
from transformers import BertTokenizer
import pandas as pd
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from decord import VideoReader
from decord import cpu, gpu
import librosa
from audiomentations import Compose,SevenBandParametricEQ, AddGaussianNoise, TimeStretch, PitchShift, Shift, Normalize, PolarityInversion, Gain, BandPassFilter, Reverse, Clip
import torch
import kornia.augmentation as K
import kornia.geometry as KG
# from vidaug import augmentors as va
from PIL import Image

import torch
import librosa
import cv2
import numpy as np
from decord import VideoReader, cpu

import torchvision.transforms.functional as TF
from PIL import Image
import cv2
# import dlib
import insightface

# Alternative face detectors - choose one
try:
    import dlib
    DLIB_AVAILABLE = True
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor("<path_to_dataset>/shape_predictor_68_face_landmarks.dat")  # Download from dlib
except ImportError:
    DLIB_AVAILABLE = False




# OpenCV Haar Cascade (always available with OpenCV)
try:
    haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    HAAR_AVAILABLE = True
except:
    HAAR_AVAILABLE = False



try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5)
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Load SCRFD face detector (default InsightFace model)
insight_detector = insightface.app.FaceAnalysis(
    name="buffalo_l",   # model pack with detector + landmarks
    providers=[ "CPUExecutionProvider"]
)
insight_detector.prepare(ctx_id=0, det_size=(640, 640))  # GPU=0, resize input to 640
# Pretrained HOG-based face detector from dlib
# dlib_detector = dlib.get_frontal_face_detector()
def detect_faces_insight(img_rgb):
    """
    Detect faces using InsightFace SCRFD.
    Returns list of (x1, y1, x2, y2, score).
    """
    faces = insight_detector.get(img_rgb)  # expects BGR or RGB np.uint8
    boxes = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)  # bbox = [x1, y1, x2, y2]
        boxes.append((x1, y1, x2, y2, float(f.det_score)))
    return boxes
def landmarks68_to_5(lm68):
    lm68 = np.asarray(lm68, dtype=np.float32).reshape(-1, 2)
    if lm68.shape[0] != 68:
        raise ValueError(f"Expected 68-point landmarks, got {lm68.shape[0]}")
    left_eye  = lm68[36:42].mean(axis=0)
    right_eye = lm68[42:48].mean(axis=0)
    nose      = lm68[30]
    mouth_l   = lm68[48]
    mouth_r   = lm68[54]
    return np.stack([left_eye, right_eye, nose, mouth_l, mouth_r], axis=0).astype(np.float32)

def create_face_mask_from_landmarks(landmarks_2d, img_shape, method='convex_hull'):
    """
    Create a binary mask from facial landmarks.
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    
    if landmarks_2d is None or len(landmarks_2d) == 0:
        return mask
    
    if method == 'convex_hull':
        # Create convex hull of all landmarks
        hull = cv2.convexHull(landmarks_2d.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    elif method == 'face_contour':
        # If using MediaPipe landmarks, use specific face contour indices
        # For now, just use convex hull as fallback
        hull = cv2.convexHull(landmarks_2d.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    
    return mask

# Pretrained HOG-based face detector from dlib
# dlib_detector = dlib.get_frontal_face_detector()

# def detect_faces_dlib(img_rgb):
#     """
#     Detect faces in an RGB image using dlib.
#     Returns a list of (x1, y1, x2, y2) boxes.
#     """
#     dets = dlib_detector(img_rgb, 1)  # upsample=1 for better small face detection
#     boxes = []
#     for d in dets:
#         x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
#         boxes.append((x1, y1, x2, y2))
#     return boxes

def detect_faces_with_landmarks(img_rgb, method='mediapipe'):
    """
    Detect faces and extract landmarks using different methods.
    Returns: (bbox, landmarks_2d) where bbox is (x, y, w, h) and landmarks_2d is array of (x, y) points
    """
    h, w = img_rgb.shape[:2]
    
    if method == 'mediapipe' and MEDIAPIPE_AVAILABLE:
        # Convert RGB to BGR for MediaPipe
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(img_rgb)  # MediaPipe expects RGB
        
        if results.multi_face_landmarks:
            # Get the first (largest) face
            face_landmarks = results.multi_face_landmarks[0]
            landmarks_2d = []
            
            for lm in face_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks_2d.append([x, y])
            
            landmarks_2d = np.array(landmarks_2d)
            
            # Calculate bounding box from landmarks
            x_min, y_min = np.min(landmarks_2d, axis=0)
            x_max, y_max = np.max(landmarks_2d, axis=0)
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            return bbox, landmarks_2d
    
    elif method == 'dlib' and DLIB_AVAILABLE:
        # Convert to grayscale for dlib
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = dlib_detector(gray)
        
        if len(faces) > 0:
            face = faces[0]  # Take first face
            bbox = (face.left(), face.top(), face.width(), face.height())
            
            # Get landmarks
            landmarks = dlib_predictor(gray, face)
            landmarks_2d = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            return bbox, landmarks_2d
    
    elif method == 'face_recognition' and FACE_RECOGNITION_AVAILABLE:
        face_locations = face_recognition.face_locations(img_rgb, model="hog")
        face_landmarks_list = face_recognition.face_landmarks(img_rgb)
        
        if face_locations and face_landmarks_list:
            # Convert face_recognition format (top, right, bottom, left) to (x, y, w, h)
            top, right, bottom, left = face_locations[0]
            bbox = (left, top, right - left, bottom - top)
            
            # Extract landmarks
            landmarks_dict = face_landmarks_list[0]
            landmarks_2d = []
            for feature_points in landmarks_dict.values():
                landmarks_2d.extend(feature_points)
            landmarks_2d = np.array(landmarks_2d)
            print(landmarks_2d)
            return bbox, landmarks_2d
    
    elif method == 'haar' and HAAR_AVAILABLE:
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = haar_face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Take first face
            bbox = (x, y, w, h)
            
            # Generate simple landmarks (just corners and center)
            landmarks_2d = np.array([
                [x, y], [x + w, y], [x + w, y + h], [x, y + h],  # corners
                [x + w//2, y + h//2]  # center
            ])
            
            return bbox, landmarks_2d
    
    return None, None


def as_rgb_uint8_np(img):
    """
    Accepts: torch.Tensor [C,H,W] or [H,W,C], PIL.Image, or np.ndarray.
    Returns: np.ndarray [H,W,3] uint8 in RGB.
    """
    # torch.Tensor
    if isinstance(img, torch.Tensor):
        x = img.detach().cpu()
        if x.dim() == 4:  # [B,C,H,W] -> take first
            x = x[0]
        if x.dim() == 3 and x.shape[0] in (1,3):  # CHW -> HWC
            x = x.permute(1,2,0)
        x = x.numpy()
        if x.dtype != np.uint8:
            # assume [0,1] or float range => clamp and scale
            x = np.clip(x, 0, 1) * 255.0
            x = x.astype(np.uint8)
        return x

    # PIL.Image
    if isinstance(img, Image.Image):
        x = np.array(img)  # already HWC, uint8 or not
        if x.dtype != np.uint8:
            x = x.astype(np.uint8)
        # ensure RGB (handle L/LA/RGBA)
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        elif x.shape[2] == 4:
            x = cv2.cvtColor(x, cv2.COLOR_RGBA2RGB)
        return x

    # numpy
    if isinstance(img, np.ndarray):
        x = img
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    raise TypeError(f"Unsupported image type for conversion: {type(img)}")


def as_bgr_uint8_np(img):
    """RGB -> BGR uint8 HWC."""
    rgb = as_rgb_uint8_np(img)
    # If it already looks BGR (you’re not sure), converting RGB->BGR twice is harmlessly symmetric.
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# Make Yunet path configurable to avoid embedding local paths
YUNET_MODEL = "<path_to_dataset>/face_detection_yunet_2023mar.onnx"
yunet = cv2.FaceDetectorYN.create(
    model=YUNET_MODEL,  # download from OpenCV zoo
    config='',
    input_size=(320, 320),  # small size = fast
    score_threshold=0.5, nms_threshold=0.3, top_k=5000
)
def detect_faces_yunet( img_any):
    """
    yunet: cv2.FaceDetectorYN instance
    img_any: torch tensor / PIL / np
    Returns: list of (x1, y1, x2, y2, score)
    """
    bgr = as_bgr_uint8_np(img_any)
    h, w = bgr.shape[:2]
    yunet.setInputSize((w, h))

    # OpenCV 4.x returns either dets or (ok, dets) depending on build – handle both:
    out = yunet.detect(bgr)
    dets = out[1] if isinstance(out, tuple) else out

    boxes = []
    if dets is not None and len(dets) > 0:
        dets = np.array(dets)
        # YuNet rows: [x, y, w, h, score, ... 10 landmark values]
        for d in dets:
            x, y, ww, hh, s = d[:5]
            x1, y1, x2, y2 = int(x), int(y), int(x + ww), int(y + hh)
            boxes.append((x1, y1, x2, y2, float(s)))
    return boxes

def img_align_crop(img, landmark, outsize=(112, 112), scale=1.3, mask=None):
    """
    landmark: (68,2) from dlib OR (5,2)
    outsize: (H, W)
    """
    # ===== ADD INPUT VALIDATION =====
    if img is None:
        raise ValueError("Input image is None")
    
    if not isinstance(img, np.ndarray):
        raise ValueError(f"Input image must be numpy array, got {type(img)}")
    
    if img.size == 0:
        raise ValueError("Input image is empty (size=0)")
    
    if len(img.shape) < 2:
        raise ValueError(f"Input image must have at least 2 dimensions, got shape: {img.shape}")
    
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError(f"Input image has zero dimensions: {img.shape}")
    
    # Check if image has valid data type
    if img.dtype not in [np.uint8, np.float32, np.float64]:
        raise ValueError(f"Unsupported image dtype: {img.dtype}")
    
    # Ensure image is uint8 for OpenCV operations
    if img.dtype != np.uint8:
        if img.max() <= 1.0:  # Assume [0,1] range
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    
    # ===== ORIGINAL LANDMARK PROCESSING =====
    if landmark is None:
        raise ValueError("landmark is required")
    lm = np.asarray(landmark, dtype=np.float32).reshape(-1, 2)
    if lm.shape[0] == 68:
        src = landmarks68_to_5(lm)
    elif lm.shape[0] == 5:
        src = lm
    else:
        raise ValueError(f"Expected 5 or 68 landmarks, got {lm.shape[0]}")

    # 5-pt reference for 112x112, with the common +8 x-shift
    ref_w, ref_h = 112, 112
    dst = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    dst[:, 0] += 8.0

    out_h, out_w = int(outsize[0]), int(outsize[1])

    # scale template to requested output size (x with width, y with height)
    dst[:, 0] *= (out_w / ref_w)
    dst[:, 1] *= (out_h / ref_h)

    # margin via scale (>1 adds borders)
    margin = max(scale, 1.0) - 1.0
    x_margin = out_w * margin / 2.0
    y_margin = out_h * margin / 2.0
    dst[:, 0] = (dst[:, 0] + x_margin) * (out_w / (out_w + 2 * x_margin))
    dst[:, 1] = (dst[:, 1] + y_margin) * (out_h / (out_h + 2 * y_margin))

    # similarity transform
    tform = trans.SimilarityTransform()
    if not tform.estimate(src, dst):
        raise RuntimeError("SimilarityTransform estimate failed")
    M = tform.params[0:2, :]  # 2x3

    # ===== SAFE WARP AFFINE =====
    try:
        aligned = cv2.warpAffine(img, M, (out_w, out_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0,0,0))
    except cv2.error as e:
        print(f"OpenCV warpAffine error: {e}")
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Transform matrix shape: {M.shape}")
        print(f"Output size: ({out_w}, {out_h})")
        raise
    
    if mask is not None:
        try:
            mask_np = to_cv2_array(mask) if hasattr(mask, 'shape') else mask
            # for masks use nearest to avoid gray edges
            warped_mask = cv2.warpAffine(mask_np, M, (out_w, out_h), 
                                       flags=cv2.INTER_NEAREST, 
                                       borderMode=cv2.BORDER_CONSTANT)
            return aligned, warped_mask
        except Exception as e:
            print(f"Error warping mask: {e}")
            return aligned, None
    
    return aligned


def to_pil(tensor):
    # Handle [B,C,H,W] or [C,H,W]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)   # drop batch
    if tensor.dtype != torch.uint8:
        tensor = (tensor.clamp(0,1) * 255).to(torch.uint8)
    return TF.to_pil_image(tensor)

def save_video_tensor(video_tensor, save_path, fps=24):
    """
    Save a video tensor [T, C, H, W] as .mp4 using OpenCV.
    Assumes video_tensor is in [0, 1] range.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    T, C, H, W = video_tensor.shape
    assert C == 3, "Expected 3 channels (RGB)"

    out = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (W, H)
    )

    for frame in video_tensor:
        frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # [H, W, C]
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"[Saved] Video to {save_path}")

audio_augmentation = Compose([
    # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    # TimeStretch(min_rate=0.2, max_rate=2.25, p=0.5),
    Shift(p=0.5),
    SevenBandParametricEQ(),
    Normalize(p=0.5),
    PolarityInversion(p=0.5),
    BandPassFilter(min_center_freq=200.0, max_center_freq=3000.0, p=0.5),  # Focus on human speech frequencies
    # Reverse(p=0.2),                                             # Reverse audio (synthetic often sounds unnatural)
    # Clip(min_percentile_threshold=0, max_percentile_threshold=20, p=0.3)    # Clip to simulate low-quality recordings
])

def sample_random_3_seconds_range(num_frames: int, fps: int = 24, min_frames: int = 300, max_frames: int = 1024):
    segment_length = 3 * fps  # 3 seconds of video

    if num_frames < min_frames or num_frames < segment_length:
        return None, None  # Not enough frames

    usable_frames = min(num_frames, max_frames)
    start_max = usable_frames - segment_length
    start_idx = random.randint(0, start_max)
    return start_idx, start_idx + segment_length


def pad_random(x: np.ndarray, max_len: int = 64000):
    x_len = x.shape[0]
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))
    return pad_random(padded_x, max_len)

def read_video_with_audio(path):
    container = av.open(path)
    video_frames = []
    audio_frames = []

    stream = container.streams.video[0]
    for frame in container.decode(stream):
        img = frame.to_image()  # Convert frame to PIL Image
        img_tensor = transforms.functional.to_tensor(img)
        video_frames.append(img_tensor)
        if len(video_frames) >= 40:
            break

    if container.streams.audio:
        audio_stream = container.streams.audio[0]
        for frame in container.decode(audio_stream):
            audio_data = frame.to_ndarray()
            audio_frames.append(audio_data)

    video_tensor = torch.stack(video_frames) if video_frames else None
    audio_tensor = np.concatenate(audio_frames, axis=1) if audio_frames else None
    # del 
    return video_tensor, torch.tensor(audio_tensor, dtype=torch.float32) if audio_tensor is not None else None

def read_video_decord(path):
    vr = VideoReader(path, ctx=cpu(0))  # Use gpu(0) for GPU
    video = vr.get_batch(range(40)).asnumpy()  # Get first 40 frames
    video = torch.tensor(video).permute(0, 3, 1, 2)  # .float() / 255.0  # Convert to torch tensor and normalize
    return video

def extract_audio_with_ffmpeg(path, output_format='wav', sample_rate=16000):
    # Temporary output file
    temp_audio = 'temp_audio.wav'
    # Command to extract audio
    command = ['ffmpeg', '-i', path, '-ar', str(sample_rate), '-ac', '1', temp_audio, '-y']
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Load the processed audio file
    audio, sr = librosa.load(temp_audio, sr=sample_rate)
    audio_tensor = torch.tensor(audio).float()

    # Delete the temporary audio file after reading
    os.remove(temp_audio)

    return audio_tensor

# def read_video(path: str):
#     video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
#     video = video.permute(0, 3, 1, 2) / 255.0
#     audio = audio.permute(1, 0)
#     return video, audio, info

# def read_video(path: str):
#     """
#     Read video frames using Decord and audio via FFmpeg (extract_audio_with_ffmpeg).
#     Returns:
#     -------
#     video_tensor: torch.Tensor 
#         Shape: (T, C, H, W), where T is #frames.
#     audio_tensor: torch.Tensor 
#         Shape: (num_samples,), mono audio signal.
#     info: dict 
#         Contains fps info for video and audio.
#     """
#     # 1. Read VIDEO with Decord
#     try:
#         vr = VideoReader(path, ctx=cpu(0))
#         num_frames = len(vr)               # total frames in the video
#         frames_array = vr.get_batch(range(num_frames)).asnumpy()  # shape: (T, H, W, C)
#         # Convert to Torch Tensor and reorder dimensions to (T, C, H, W)
#         video_tensor = torch.tensor(frames_array).permute(0, 3, 1, 2) / 255.0
#         video_fps = vr.get_avg_fps()
#     except Exception as e:
#         print(f"Error reading video from {path} with Decord: {e}")
#         # Fallback: empty or dummy data if needed
#         video_tensor = torch.empty(0)
#         video_fps = 30  # default/fallback

#     # 2. Read AUDIO with FFmpeg helper
#     #    This uses your existing function extract_audio_with_ffmpeg
#     # audio_tensor = extract_audio_with_ffmpeg(path, sample_rate=16000)  # shape: (num_samples,)
#     audio, sr = librosa.load(path.replace("mp4","wav"), sr=16000)
#     # Convert to PyTorch tensor
#     audio_tensor = torch.tensor(audio, dtype=torch.float32)
#     # 3. Prepare info dictionary
#     info = {
#         "video_fps": video_fps, 
#         "audio_fps": 16000      # you set sample_rate=16000 in extract_audio_with_ffmpeg
#     }
#     return video_tensor, audio_tensor, info


def read_video(path: str, desired_num_frames, validation):
    """
    Read video frames using Decord, apply histogram equalization to each frame,
    and read audio with librosa (or your FFmpeg-based method).
    Returns:
    -------
    video_tensor: torch.Tensor 
        Shape: (T, C, H, W), where T is #frames, with histogram-equalized frames.
    audio_tensor: torch.Tensor 
        Shape: (num_samples,), mono audio signal.
    info: dict 
        Contains fps info for video and audio.
    """
    # 1. Read VIDEO with Decord
    try:
        vr = VideoReader(path, ctx=cpu(0))
        # num_frames = desired_num_frames#len(vr)  # total frames in the video
        # # print(num_frames)
        # # shape: (T, H, W, C) in RGB order
        # frames_array = vr.get_batch(range(desired_num_frames)).asnumpy()  
        # video_fps = vr.get_avg_fps()
        total_frames = len(vr)
        start_idx, end_idx = sample_random_3_seconds_range(total_frames, fps=24)
        # print(start_idx, end_idx)
        if start_idx == None or validation:
            start_idx=0
            end_idx=desired_num_frames

        frames_array = vr.get_batch(range(start_idx, end_idx)).asnumpy()
        video_fps = vr.get_avg_fps()
    except Exception as e:
        print(f"Error reading video from {path} with Decord: {e}")
        frames_array = np.empty((0, 0, 0, 0), dtype=np.uint8)
        video_fps = 30  # fallback fps

    # 2. Apply histogram equalization on each frame
    #    We'll process frames in RGB -> YCrCb -> equalize Y -> back to RGB
    eq_frames = []
    # output_dir="./output_dir"
    for i in range(frames_array.shape[0]):
        frame_rgb = frames_array[i]  # shape: (H, W, C), RGB

        # Convert RGB -> YCrCb
        frame_ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
        # Equalize only the Y (luminance) channel
        frame_ycrcb[..., 0] = cv2.equalizeHist(frame_ycrcb[..., 0])
        # Convert back to RGB
        frame_eq = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2RGB)
        # frame_filename = os.path.join(output_dir, f"frame_{i:04d}.png")
        # cv2.imwrite(frame_filename, cv2.cvtColor(frame_eq, cv2.COLOR_RGB2BGR))
        # print(f"Saved equalized frame: {frame_filename}")
        eq_frames.append(frame_eq)

    # Convert list -> NumPy -> Torch
    if len(eq_frames) > 0:
        frames_array_eq = np.stack(eq_frames, axis=0)  # (T, H, W, C)
        video_tensor = torch.tensor(frames_array_eq).permute(0, 3, 1, 2) / 255.0
    else:
        # Empty fallback
        video_tensor = torch.empty((0, 3, 0, 0))

    # 3. Read AUDIO (example using librosa)
    #    If you have a .wav audio file with the same name, just replace extension
    #    or adapt to your existing audio-extraction function
    # audio_path = path.replace(".mp4", ".wav")
    # audio_path has been anonymized; set SAMPLE_AUDIO env var if you need a sample file
    # audio_path = os.environ.get("SAMPLE_AUDIO", "")
    # try:
    #     audio, sr = librosa.load(audio_path, sr=16000)
    #     audio_tensor = torch.tensor(audio, dtype=torch.float32)
    # except Exception as e:
    #     print(f"Error reading audio from {audio_path} with librosa: {e}")
    #     audio_tensor = torch.empty(0)
    audio_tensor = torch.empty(0)
    # 4. Prepare info dictionary
    info = {
        "video_fps": video_fps, 
        "audio_fps": 16000  # or your chosen sample rate
    }
    return video_tensor, audio_tensor, info



def temporal_jitter(video: torch.Tensor,
                    max_drop: int = 3,
                    max_dup: int = 3) -> torch.Tensor:
    """
    Randomly drop or duplicate a few frames – mimics stutter / missing frames.
    video: [T, C, H, W]
    """
    T = video.shape[0]
    num_drop = random.randint(0, max_drop)
    if num_drop:
        keep_idx = sorted(random.sample(range(T), k=T - num_drop))
        video = video[keep_idx]

    num_dup = random.randint(0, max_dup)
    if num_dup:
        dup_idx = random.choices(range(video.shape[0]), k=num_dup)
        dup_frames = video[dup_idx]
        video = torch.cat([video, dup_frames], dim=0)

    # ensure original length by trimming / padding zeros
    if video.shape[0] < T:
        pad = torch.zeros((T - video.shape[0], *video.shape[1:]),
                        dtype=video.dtype, device=video.device)
        video = torch.cat([video, pad], dim=0)
    elif video.shape[0] > T:
        video = video[:T]

    return video
    
def extract_mfcc(audio, sample_rate=16000, n_mfcc=40, hop_length=512, n_fft=2048):
    """
    Extract MFCC features from audio.

    Parameters:
    - audio (np.ndarray): Audio time series.
    - sample_rate (int): Sampling rate of the audio.
    - n_mfcc (int): Number of MFCCs to return.
    - hop_length (int): Number of samples between successive frames.
    - n_fft (int): Length of the FFT window.

    Returns:
    - mfccs (np.ndarray): MFCC feature matrix.
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    mfccs = librosa.power_to_db(mfccs, ref=np.max)  # Convert to log scale (dB)
    return mfccs
def equalize_frames(frames_np: np.ndarray) -> torch.Tensor:
    """Histogram‑equalise luminance channel of a (T, H, W, C) RGB numpy array."""
    eq = []
    for f in frames_np:
        ycrcb = cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)
        ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
        eq.append(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB))
    eq = np.stack(eq, 0)  # (T, H, W, C)
    return torch.from_numpy(eq).permute(0, 3, 1, 2).float() / 255  # (T, C, H, W)



def normalize_landmarks_standardize(landmarks):
 
    mean = np.mean(landmarks, axis=0)
    std = np.std(landmarks, axis=0)
    std[std == 0] = 1

    return (landmarks - mean) / std

def get_image_path(root, method, pair_filename, compression):

    return os.path.join(
        root,
        # "video_data",
        # "data_raw",
        "manipulated_sequences",
        method,
        compression,
        "faces",
        pair_filename
    )



def get_image_path_effort(root, method, pair_filename, compression):
    return os.path.join(
        root,
        "manipulated_sequences",
        method,
        compression,
        "frames",
        pair_filename
    )

def get_real_path_effort(root, video_id, compression):
    return os.path.join(
        root,
        "original_sequences",
        "youtube",
        compression,
        "frames",
        f"{video_id}"
    )

def check_and_add(path, path_list):
    if os.path.exists(path):
        path_list.append(path)

def get_real_path(root, video_id, compression):
    return os.path.join(
        root,
        # "video_data",
        # "data_raw",
        "original_sequences",
        "youtube",
        compression,
        "faces",
        f"{video_id}"
    )
    import torch
import torch.nn as nn
from typing import Dict, Tuple

class RandomBandOrBoxErase(nn.Module):
    """
    Erase a band (top/middle/bottom/left/right) or a box (center/random) with
    configurable probabilities. Works for BCHW and BCTHW.
    - lock_t=True: for 5D inputs, apply the SAME region to every frame T of each sample.
    - same_on_batch=True: sample ONE region and apply it to every sample in the batch.
    """
    def __init__(
        self,
        p: float = 1.0,                          # overall probability to apply any erase
        band_h_range: Tuple[int, int] = (36, 50),  # horizontal band height range (px)
        band_w_range: Tuple[int, int] = (36, 50),  # vertical band width range (px)
        square_size_range: Tuple[int, int] = (48, 52),  # ~50x50
        strategy_probs: Dict[str, float] = None,  # probs over {top,middle,bottom,left,right,center_square,random_square}
        value: float = 0.0,                      # fill value
        lock_t: bool = True,                     # for 5D: same region across time
        same_on_batch: bool = False              # sample one region for entire batch
    ):
        super().__init__()
        self.p = float(p)
        self.band_h_range = band_h_range
        self.band_w_range = band_w_range
        self.square_size_range = square_size_range
        self.value = value
        self.lock_t = lock_t
        self.same_on_batch = same_on_batch

        # default: equal mass over bands; small mass to random square
        default_probs = {
            "top": 0.18, "middle": 0.18, "bottom": 0.18,
            "left": 0.18, "right": 0.18,
            "center_square": 0.05, "random_square": 0.05
        }
        self.strategy_probs = strategy_probs or default_probs
        self._names = list(self.strategy_probs.keys())
        w = torch.tensor([self.strategy_probs[k] for k in self._names], dtype=torch.float)
        if w.sum() <= 0:
            raise ValueError("strategy_probs must have positive total mass.")
        self._weights = (w / w.sum())

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        if x.dim() == 4:          # (B, C, H, W)
            return self._apply_bchw(x)
        elif x.dim() == 5:        # (B, C, T, H, W)
            return self._apply_bcthw(x)
        else:
            return x

    def _apply_bchw(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out = x.clone()
        device = x.device

        # Optionally sample one region for the whole batch
        if self.same_on_batch:
            y0, y1, x0, x1 = self._sample_region(H, W, device)
            out[:, :, y0:y1, x0:x1] = self.value
            return out

        # Otherwise, sample per sample
        for b in range(B):
            y0, y1, x0, x1 = self._sample_region(H, W, device)
            out[b, :, y0:y1, x0:x1] = self.value
        return out

    def _apply_bcthw(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        out = x.clone()
        device = x.device

        if self.same_on_batch:
            # One region for entire batch; apply to all frames (locks time implicitly)
            y0, y1, x0, x1 = self._sample_region(H, W, device)
            out[:, :, :, y0:y1, x0:x1] = self.value
            return out

        # Per-sample region
        for b in range(B):
            if self.lock_t:
                # Same region across time for sample b
                y0, y1, x0, x1 = self._sample_region(H, W, device)
                out[b, :, :, y0:y1, x0:x1] = self.value
            else:
                # Different region for each frame (more jittery)
                for t in range(T):
                    y0, y1, x0, x1 = self._sample_region(H, W, device)
                    out[b, :, t, y0:y1, x0:x1] = self.value
        return out

    def _sample_region(self, H: int, W: int, device) -> Tuple[int, int, int, int]:
        # pick a strategy by weights
        idx = torch.multinomial(self._weights.to(device), num_samples=1).item()
        name = self._names[idx]

        def clamp_int(a, lo, hi): return int(max(lo, min(hi, a)))

        if name in ("top", "middle", "bottom"):
            h = int(torch.empty(1, device=device).uniform_(*self.band_h_range).item())
            h = clamp_int(h, 1, H)
            if name == "top":
                y0 = 0
            elif name == "middle":
                y0 = clamp_int((H - h) // 2, 0, max(0, H - h))
            else:  # bottom
                y0 = clamp_int(H - h, 0, max(0, H - h))
            y1 = y0 + h
            x0, x1 = 0, W
            return y0, y1, x0, x1

        if name in ("left", "right"):
            w = int(torch.empty(1, device=device).uniform_(*self.band_w_range).item())
            w = clamp_int(w, 1, W)
            if name == "left":
                x0 = 0
            else:  # right
                x0 = clamp_int(W - w, 0, max(0, W - w))
            x1 = x0 + w
            y0, y1 = 0, H
            return y0, y1, x0, x1

        if name == "center_square":
            s = int(torch.empty(1, device=device).uniform_(*self.square_size_range).item())
            s = clamp_int(s, 1, min(H, W))
            y0 = clamp_int((H - s) // 2, 0, max(0, H - s))
            x0 = clamp_int((W - s) // 2, 0, max(0, W - s))
            return y0, y0 + s, x0, x0 + s

        # "random_square"
        s = int(torch.empty(1, device=device).uniform_(*self.square_size_range).item())
        s = clamp_int(s, 1, min(H, W))
        y0 = torch.randint(0, max(1, H - s + 1), (1,), device=device).item()
        x0 = torch.randint(0, max(1, W - s + 1), (1,), device=device).item()
        return y0, y0 + s, x0, x0 + s

def _tensor_rgb_to_bgr_uint8(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: [C,H,W] in [0,1] or uint8, or [1,C,H,W].
    returns BGR uint8 HxWx3 for OpenCV.
    """
    if img_t.dim() == 4:
        img_t = img_t[0]
    if img_t.dtype != torch.uint8:
        img_t = (img_t.clamp(0,1) * 255).to(torch.uint8)
    # CHW -> HWC RGB -> BGR
    return cv2.cvtColor(img_t.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR)

def _detect_yunet_boxes_bgr(bgr: np.ndarray):
    """
    Returns a list of (x, y, w, h, score). Empty list if none.
    Assumes a global/module-level `yunet` already created.
    """
    H, W = bgr.shape[:2]
    yunet.setInputSize((W, H))
    out = yunet.detect(bgr)
    dets = out[1] if isinstance(out, tuple) else out
    if dets is None:
        return []
    dets = np.array(dets, dtype=np.float32)  # N x 15  [x,y,w,h,score, 10 lm coords]
    return [(float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])) for d in dets]



def _expand_box(xywh, im_w, im_h,
                expand_left=0.20, expand_top=0.40,
                expand_right=0.20, expand_bottom=0.25):
    """
    Expand (x,y,w,h) to include more context around the face.
    Expansions are fractional relative to the width/height of the box.
    Returns clamped xyxy ints.
    """
    x, y, w, h = xywh

    # Expand each side by fraction of w or h
    x0 = x - expand_left * w
    y0 = y - expand_top * h
    x1 = (x + w) + expand_right * w
    y1 = (y + h) + expand_bottom * h

    # Clamp to image boundaries
    x0 = max(0, int(np.floor(x0)))
    y0 = max(0, int(np.floor(y0)))
    x1 = min(im_w, int(np.ceil(x1)))
    y1 = min(im_h, int(np.ceil(y1)))

    # Fallback if expansion went invalid
    if x1 <= x0 or y1 <= y0:
        return int(x), int(y), int(x + w), int(y + h)
    return x0, y0, x1, y1


def _shrink_box_anisotropic(xywh, im_w, im_h,
                            shrink_left=0.12, shrink_top=0.35,
                            shrink_right=0.12, shrink_bottom=0.10):
    """
    Shrink (x,y,w,h) more at the top to remove hair. Returns clamped xyxy ints.
    """
    x, y, w, h = xywh
    x0 = x + shrink_left  * w
    y0 = y + shrink_top   * h
    x1 = (x + w) - shrink_right  * w
    y1 = (y + h) - shrink_bottom * h

    x0 = max(0, int(np.floor(x0))); y0 = max(0, int(np.floor(y0)))
    x1 = min(im_w, int(np.ceil(x1))); y1 = min(im_h, int(np.ceil(y1)))
    if x1 <= x0 or y1 <= y0:  # fallback to original xyxy
        return int(x), int(y), int(x + w), int(y + h)
    return x0, y0, x1, y1

def _crop_tensor_xyxy(img_t: torch.Tensor, box_xyxy):
    """
    img_t: [C,H,W] or [1,C,H,W]; box: (x0,y0,x1,y1). Returns [C,h,w].
    """
    if img_t.dim() == 4:  # [1,C,H,W] -> [C,H,W]
        img_t = img_t[0]
    C, H, W = img_t.shape
    x0, y0, x1, y1 = map(int, box_xyxy)
    x0 = max(0, min(W-1, x0)); x1 = max(1, min(W, x1))
    y0 = max(0, min(H-1, y0)); y1 = max(1, min(H, y1))
    if x1 <= x0 or y1 <= y0:  # empty -> return original
        return img_t
    return img_t[:, y0:y1, x0:x1]

class FaceswapImagesDatasetAligned(Dataset):
    """
    Dataset class for the SAMMD2024 dataset.
    """
    def __init__(self, base_dir, partition="train",take_datasets="1,2,3,4,5,6", 
        clip_len_sec: int = 1,
        clip_stride_sec: int | None = 1,
        fps: int = 16,
    max_len=64000, frame_rate=1, n_mfcc=40):
        assert partition in ["train", "dev", "test"], "Invalid partition. Must be one of ['train', 'dev', 'test']"
        self.base_dir = base_dir
        self.data_labels = take_datasets
        self.partition = partition
        self.max_len = max_len
        self.fps = fps
        self.clip_len = clip_len_sec * fps
        self.clip_stride = (clip_stride_sec or clip_len_sec) * fps

        MAX_FRAMES=8
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.frame_rate = frame_rate  # Number of frames to extract per second
        self.real_dir = os.path.join(base_dir, f"{partition}/real")
        print(f"{self.base_dir}deepspeak/dataset/")
        self.face_detector = fasterrcnn_resnet50_fpn(pretrained=True)
        self.face_detector.eval()  # Set model to evaluation mode
        self.fake_dir = os.path.join(base_dir, f"{partition}/fake")
        video_files = {}
        if partition=="train":
            self.is_train=True
        else:
            self.is_train=False
        # ---------------------------------------------------------------
        # AUDIO pipeline  
        # ---------------------------------------------------------------
        self.audio_aug = Compose([
            SevenBandParametricEQ(p=0.9),
            AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.015, p=0.9),
            PitchShift(min_semitones=-2, max_semitones=+2, p=0.9),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.9),
            # Shift(min_fraction=-0.2, max_fraction=0.2, p=0.6),
            Shift(p=0.5),
            # small loudness jitter
            Gain(min_gain_db=-6, max_gain_db=6, p=0.9),
            Normalize(p=0.6),
            PolarityInversion(p=0.9),
            BandPassFilter(min_center_freq=150.0, max_center_freq=3500.0, p=0.9),
        ])

        self.real_aug = K.AugmentationSequential(
            
            K.Resize((224, 224)),
            
            # K.RandomHorizontalFlip(p=0.9),
            # K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.1),
            # K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.2),
            # K.RandomMotionBlur(kernel_size=3, angle=15., direction=0.5,p=0.1),
            # K.RandomAffine(degrees=4,
            #                translate=(0.02, 0.02),
            #                scale=(0.9, 1.1),
            #                shear=None,
            #                p=0.1),
            # same_on_batch=True  # <- this is key
            # data_format="BCTHW"
        )
        self.fake_aug = K.AugmentationSequential(
            
            K.Resize((224, 224)),
        
            # K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.1),
            # K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.2),
            # K.RandomMotionBlur(kernel_size=3, angle=15., direction=0.5,p=0.1),
            # K.RandomAffine(degrees=4,
            #                translate=(0.02, 0.02),
            #                scale=(0.9, 1.1),
            #                shear=None,
            #                p=0.1),
            # K.RandomErasing(scale=(0.02, 0.33), ratio=(0.5, 3.6), p=0.9),      # sim. occlusion
            # same_on_batch=True  # <- this is key
            # data_format="BCTHW"
        )

        self.real_aug_ori = K.AugmentationSequential(
            
                    # K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.4),
                    # K.RandomMotionBlur(kernel_size=3, angle=15., direction=0.5,p=0.1),
                    # K.RandomErasing(scale=(0.02, 0.33), ratio=(0.5, 3.6), p=0.1),      # sim. occlusion
                    # data_format="BCTHW"
                    same_on_batch=True  # <- this is key
                )

        self.real_aug_diff = K.AugmentationSequential(
            
                    K.Resize((224, 224)),
                    # K.RandomHorizontalFlip(p=0.5),
                    # K.RandomErasing(scale=(0.4, 0.5), ratio=(0.8, 1.2), p=0.3),      # sim. occlusion
                    
                    # K.RandomErasing(scale=(0.3, 0.5), ratio=(0.8, 1.2), p=0.4),   # big squares
                    # K.RandomErasing(scale=(0.15, 0.25), ratio=(4, 8), p=0.3),     # long bands
                    #  RandomBandOrBoxErase(
                    #         p=0.25,  # usually apply an erase
                    #         band_h_range=(36, 64),
                    #         band_w_range=(36, 64),
                    #         square_size_range=(48, 56),
                    #         strategy_probs={
                    #             "top": 0.18, "middle": 0.18, "bottom": 0.18,
                    #             "left": 0.18, "right": 0.18,
                    #             "center_square": 0.05, "random_square": 0.05
                    #         },
                    #         value=0.0,     # keep 0 for diff frames; for RGB you might try 0.5 or noise
                    #         lock_t=True,   # SAME region across frames of a clip (recommended)
                    #         same_on_batch=False  # set True if you want the same mask for the whole batch
                    #     ),
                    # K.RandomAffine(degrees=4,
                    #         translate=(0.02, 0.02),
                    #         scale=(0.9, 1.1),
                    #         shear=None,
                    #         p=0.2),
                    # data_format="BCTHW"
                    same_on_batch=True  # <- this is key
                )
        # self.real_aug = K.AugmentationSequential(
        #     K.RandomHorizontalFlip(p=0.3),                        # minor flip chance
        #     K.ColorJitter(
        #         brightness=0.1, 
        #         contrast=0.1, 
        #         saturation=0.1, 
        #         hue=0.02, 
        #         p=0.3
        #     ),  
        #     K.RandomGaussianBlur((3, 3), (0.1, 0.4), p=0.15),    # very mild blur
        #     K.RandomErasing(scale=(0.01, 0.05), ratio=(0.5, 2.0), p=0.2),  # mild occlusion
        #     K.RandomMotionBlur(kernel_size=3, angle=5., direction=0.5, p=0.1),
        # )

                
        # sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
        # self.vaseq = va.Sequential([
        #     va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
        #     va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
        #     sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
        # ])

        print("started to load data")
        
              
        self.file_list = []
        
        # Initialize fake category lists
        fake_methods = {
            "Deepfakes": [],
            "Face2Face": [],
            # "FaceShifter": [],
            "FaceSwap": [],
            "NeuralTextures": []
        }
        real_paths = []

        # ────────────────────────────────────────────────────────────
        REAL=0
        FAKE=0
        cache_file = f"faceswap_ds_{partition}_with_all_compressions.json"
        
        if os.path.exists(cache_file):
            print(f"Loading file list from cache: {cache_file}")
            with open(cache_file, "r") as f:
                self.file_list = json.load(f)
                
            self.fake_image_list = []
            self.real_image_list = []
            self.label_list = []

            for item in self.file_list:
                # you can choose either real image, fake image, or both
                self.real_image_list.append(item["path"])   # real image path
                self.fake_image_list.append(item["fake_path"])   # real image path
                self.label_list.append(item["label"])  # label (0 or 1 etc.)

            self.data_dict = {
                'image': self.fake_image_list,
                'image_real': self.real_image_list,
                'label': self.label_list
            }
            print(f"Loaded {len(self.file_list)} files from cache.")
            return

        
        # Build dataset from FaceForensics++
        for compression in ["c23"]:
            print("We are loading data right now from directories")
            root = Path(f"<path_to_dataset>/dataset/FaceForensics++/")
            
            if partition == "train":
                df = pd.read_json(os.path.join(root, "train.json"))
            else:
                df = pd.read_json(os.path.join(root, "test.json"))

            for _, row in df.iterrows():
                pair1 = str(row[0]).zfill(3)
                pair2 = str(row[1]).zfill(3)

                for method in fake_methods.keys():
                    pair_filename_1 = f"{pair1}_{pair2}"
                    pair_filename_2 = f"{pair2}_{pair1}"
                    
                    # Process first pair
                    real_path = get_real_path_effort(root, pair1, compression)
                    fake_path = get_image_path_effort(root, method, pair_filename_1, compression)
                    
                    if os.path.exists(real_path) and os.path.exists(fake_path):
                        frames = 0
                        for file in os.listdir(real_path):
                            if file.endswith(".jpg") or file.endswith(".png"):
                                REAL += 1
                                frames += 1
                                if frames >= MAX_FRAMES:
                                    break
                                real_file_path = os.path.join(real_path, file)
                                fake_file_path = os.path.join(fake_path, file)
                                if os.path.exists(real_file_path) and os.path.exists(fake_file_path):
                                    self.file_list.append({
                                        "path": real_file_path,
                                        "fake_path": fake_file_path,
                                        "label": 0,
                                        "multi_label": 0,
                                        "method": method
                                    })
                    
                    # Process second pair
                    real_path = get_real_path_effort(root, pair2, compression)
                    fake_path = get_image_path_effort(root, method, pair_filename_2, compression)
                    
                    if os.path.exists(real_path) and os.path.exists(fake_path):
                        frames = 0
                        for file in os.listdir(real_path):
                            if file.endswith(".jpg") or file.endswith(".png"):
                                FAKE += 1
                                frames += 1
                                if frames >= MAX_FRAMES:
                                    break
                                real_file_path = os.path.join(real_path, file)
                                fake_file_path = os.path.join(fake_path, file)
                                if os.path.exists(real_file_path) and os.path.exists(fake_file_path):
                                    self.file_list.append({
                                        "path": real_file_path,
                                        "fake_path": fake_file_path,
                                        "label": 0,
                                        "multi_label": 0,
                                        "method": method
                                    })
        # Save built file_list to cache


        with open(cache_file, "w") as f:
            json.dump(self.file_list, f, indent=2)
       
        print(f"Total files in {partition} are {len(self.file_list)} Total Fakes: {FAKE} and TOTAL REAL: {REAL}")
        


   
    def __len__(self):
        return len(self.file_list)

    def _load_diff_clip(self, file_path: str, idx: int, st: int, en: int):
        """
        Loads frames [st, en) from file_path, applies diff augmentation per-frame,
        returns Tensor of shape (T, 3, H, W), dtype=float32 in [0,1].
        Falls back to next item on I/O errors.
        """
        frames_np = None
        try:
            vr = VideoReader(file_path, ctx=cpu(0))  # fixed: use file_path, decord.cpu
            # clamp the frame range to the actual number of frames
            frame_indices = list(range(st, min(en, len(vr))))
            if not frame_indices:
                # no frames in range -> advance to next item
                return self.__getitem__((idx + 1) % len(self))
            frames_np = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, 3), uint8
        except Exception:
            traceback.print_exc()
            return self.__getitem__((idx + 1) % len(self))

        T, H, W, C = frames_np.shape  # C should be 3

        # If shorter than desired clip_len, pad with zeros (black frames) to the right.
        if T < self.clip_len:
            pad_len = self.clip_len - T
            pad = np.zeros((pad_len, H, W, C), dtype=frames_np.dtype)
            frames_np = np.concatenate([frames_np, pad], axis=0)  # now (clip_len, H, W, 3)

        # Convert to torch and normalize to [0,1]: (T, H, W, 3) -> (T, 3, H, W)
        frames_t = torch.from_numpy(frames_np)                # uint8
        frames_t = frames_t.permute(0, 3, 1, 2).contiguous()  # (T, 3, H, W)
        frames_t = frames_t.float().div_(255.0)               # float32 in [0,1]

        # Apply per-frame "difference" augmentation if needed.
        # Assuming self.real_aug_diff expects a single (3, H, W) float tensor and returns same
        # If it can handle batches, just call it once on frames_t.
        if hasattr(self, "real_aug_diff") and callable(self.real_aug_diff):
            frames_aug = []
            for f in frames_t:  # f: (3, H, W)
                f_aug = self.real_aug_diff(f)  # keep shape (3, H, W)
                f_aug = f_aug.squeeze(0)
                # Avoid squeezing channel dims; keep (3,H,W)
                frames_aug.append(f_aug)
            diff_video = torch.stack(frames_aug, dim=0)  # (T, 3, H, W)
        else:
            diff_video = frames_t  # no-op if no augmentation

        return diff_video  # (T, 3, H, W), float32 in [0,1]


    def get_face_crop(self, frames):
        face_frames = []
        with torch.no_grad():  # Disable gradient calculation for inference
            for frame in frames:
                try:
                    # Frame shape is assumed to be (C, H, W), normalize to [0, 1]
                    # frame = frame / 255.0
                    # Add batch dimension for inference
                    frame = frame.unsqueeze(0)  # (1, C, H, W)
                    # Perform face detection
                    detections = self.face_detector(frame)[0]
                    # Process detected faces
                    for box, score in zip(detections['boxes'], detections['scores']):
                        if score >= 0.5:  # Confidence threshold for face detection
                            x1, y1, x2, y2 = map(int, box.tolist())
                            face = frame[0, :, y1:y2, x1:x2]  # Crop the face
                            face = T.Resize((224, 224))(face)  # Resize to 224x224
                            face_frames.append(face)
                            # Save the cropped face to a directory
                
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue

        # Handle case where no faces are detected
        if not face_frames:
            print("No faces detected in the frames.")
            return frames # Return a single black frame as placeholder

        # Stack detected face tensors
        return torch.stack(face_frames)
    def __getitem__(self, index):
        try:
            item = self.file_list[index]
            file_path = item["path"]
            fake_path = item["fake_path"]
            label = item["label"]

            # --- Load as NumPy ---
            real = Image.open(file_path).convert("RGB")
            fake = Image.open(fake_path).convert("RGB")
            real_np = np.array(real)
            fake_np = np.array(fake)

            # --- Detect landmarks on REAL first (fallback to FAKE) ---
            bbox, landmarks_2d = detect_faces_with_landmarks(real_np, method="dlib")
            
            if landmarks_2d is None or bbox is None:
                bbox, landmarks_2d = detect_faces_with_landmarks(fake_np, method="dlib")
            
            # --- Build a 5-point set for alignment ---
            def bbox_to_5pt(b):
                if b is None:
                    h, w = real_np.shape[:2]
                    x, y, bw, bh = int(0.1*w), int(0.1*h), int(0.8*w), int(0.8*h)
                else:
                    x, y, bw, bh = b
                le = (x + 0.35 * bw, y + 0.40 * bh)
                re = (x + 0.65 * bw, y + 0.40 * bh)
                no = (x + 0.50 * bw, y + 0.55 * bh)
                ml = (x + 0.38 * bw, y + 0.75 * bh)
                mr = (x + 0.62 * bw, y + 0.75 * bh)
                return np.array([le, re, no, ml, mr], dtype=np.float32)

            if landmarks_2d is not None and len(landmarks_2d) > 0:
                n = landmarks_2d.shape[0]
                if n == 68:
                    lm5 = landmarks68_to_5(landmarks_2d)
                elif n == 5:
                    lm5 = landmarks_2d.astype(np.float32)
                else:
                    lm5 = bbox_to_5pt(bbox)
            else:
                lm5 = bbox_to_5pt(bbox)
         

            # --- Align BOTH images to the same canonical frame ---
            # real_aligned = img_align_crop(real_np, landmark=lm5, outsize=(224, 224), scale=0.7)
            # fake_aligned = img_align_crop(fake_np, landmark=lm5, outsize=(224, 224), scale=0.7)


            # real_aligned = torch.from_numpy(real_aligned)
            # fake_aligned = torch.from_numpy(fake_aligned)
            real_pil = Image.fromarray(real_np)      # np.ndarray → PIL
            fake_pil = Image.fromarray(fake_np)

    # --- Convert to tensor and stack for SAME augmentation ---
            real_t = TF.to_tensor(TF.resize(real_pil, (224, 224))).unsqueeze(0)   # [1,C,H,W]
            fake_t = TF.to_tensor(TF.resize(fake_pil, (224, 224))).unsqueeze(0)
            pair_t = torch.cat([real_t, fake_t], dim=0)   # [2,C,H,W]
    # --- Shared augmentat`ion before alignment ---
            if self.is_train:
                aug_seq = K.AugmentationSequential(
                    K.Resize((224, 224)),  # keep target size consistent
                    K.RandomHorizontalFlip(p=0.5),
                    K.RandomAffine(
                        degrees=4, 
                        translate=(0.02, 0.02),
                        scale=(0.9, 1.1),
                        shear=None,
                        p=0.2
                    ),
                    K.ColorJitter(0.2, 0.2, 0.2, 0.1),
                    same_on_batch=True,   # <- ensures both get the SAME aug
                    data_keys=["input"]
                )
                pair_t = aug_seq(pair_t)

            real_aligned = pair_t[0].permute(1, 2, 0).cpu().numpy() * 255
            fake_aligned = pair_t[1].permute(1, 2, 0).cpu().numpy() * 255
            real_aligned = real_aligned.astype(np.uint8)
            fake_aligned = fake_aligned.astype(np.uint8)
            # --- CORRECTED: Compute difference and extract meaningful regions ---
            # real_aligned = cv2.cvtColor(real_aligned, cv2.COLOR_BGR2RGB) if real_aligned.shape[2] == 3 else real_aligned
            # fake_aligned = cv2.cvtColor(fake_aligned, cv2.COLOR_BGR2RGB) if fake_aligned.shape[2] == 3 else fake_aligned

            # 1. Compute per-pixel difference (absolute)
            diff_rgb = cv2.absdiff(real_aligned, fake_aligned)
            
            # 2. Convert difference to grayscale for thresholding
                        
            # 3. Convert diff to grayscale
            diff_gray = cv2.cvtColor(diff_rgb, cv2.COLOR_RGB2GRAY)

            # 4. Adaptive threshold instead of fixed 10
            _, diff_mask = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            
            
            # # 4. Optional: Apply morphological operations to clean up the mask
            # kernel = np.ones((3,3), np.uint8)
            # diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
            # diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
                        
            # 5. Morphological clean-up
            kernel = np.ones((3,3), np.uint8)
            diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
            diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
            
            # 5. Extract RGB pixels from the FAKE image where differences exist
            # This gives you the actual fake pixels in regions that differ from real
            pixels_rgb = np.zeros_like(fake_aligned)
            pixels_rgb[diff_mask > 0] = fake_aligned[diff_mask > 0]
            
            # Alternative: You might want to show the difference itself with enhancement
            # Enhanced difference for visualization
            # diff_enhanced = diff_rgb.astype(np.float32)
            # diff_enhanced = np.clip(diff_enhanced * 3.0, 0, 255).astype(np.uint8)  # Amplify differences
            
            # 6. Extract fake-only regions
            pixels_rgb = np.zeros_like(fake_aligned)
            pixels_rgb[diff_mask > 0] = fake_aligned[diff_mask > 0]

            # 7. (Optional) Enhanced difference only for visualization
            diff_enhanced = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
            # --- Convert to tensors ---
            MEAN = [0.48145466, 0.4578275, 0.40821073]
            STD = [0.26862954, 0.26130258, 0.27577711]

            real_t = TF.to_tensor(real_aligned)
            fake_t = TF.to_tensor(fake_aligned)
            diff_t = TF.to_tensor(diff_rgb)  # Use enhanced difference
            pixels_rgb_t = TF.to_tensor(pixels_rgb)

            # Apply augmentations if in training mode
            # if self.is_train:
            #     real_t_aug = self.real_aug(real_t.unsqueeze(0)).squeeze(0)
            #     fake_t_aug = self.fake_aug(fake_t.unsqueeze(0)).squeeze(0)
            #     diff_t_aug = self.real_aug_diff(diff_t.unsqueeze(0)).squeeze(0)
                
            #     real_t = real_t_aug
            #     fake_t = fake_t_aug
            #     diff_t = diff_t_aug

            # Normalize real and fake with ImageNet stats
            real_t = TF.normalize(real_t, mean=MEAN, std=STD)
            fake_t = TF.normalize(fake_t, mean=MEAN, std=STD)
            
            # For pixels_rgb, only normalize if it has content
            # if pixels_rgb.max() > 0:
            #     pixels_rgb_t = TF.normalize(pixels_rgb_t, mean=MEAN, std=STD)
            # Otherwise keep it as zeros (black)
            
            # diff_t can remain unnormalized since it's a difference visualization
            
            return real_t, fake_t, diff_t, pixels_rgb_t, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"[Error] in __getitem__ at index {index}: {e}")
            import traceback
            traceback.print_exc()
            return self.__getitem__((index + 1) % len(self))

            
    # def __getitem__(self, index):
    #     try:
    #         item = self.file_list[index]
    #         file_path = item["path"]
    #         fake_path = item["fake_path"]
    #         label = item["label"]

    #         # Load PIL -> tensor [C,H,W] in [0,1]
    #         real_pil = Image.open(file_path).convert("RGB")
    #         fake_pil = Image.open(fake_path).convert("RGB")

    #         # ------------------- FACE DETECTION ------------------- #
    #         real_np = np.array(real_pil)
    #         image_bgr = cv2.cvtColor(real_np, cv2.COLOR_RGB2BGR)
    #         boxes = detect_faces_insight(image_bgr)

    #         if len(boxes) > 0:
    #             # pick largest face
    #             x1, y1, x2, y2, _ = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    #             # image_pil = image_pil.crop((x1, y1, x2, y2))
    #             real_pil = real_pil.crop((x1, y1, x2, y2))
    #             fake_pil = fake_pil.crop((x1, y1, x2, y2))  # same crop

    #         # real_bgr = cv2.cvtColor(real_np, cv2.COLOR_RGB2BGR)
    #         # boxes = _detect_yunet_boxes_bgr(real_bgr)
    #         # boxes = detect_faces_dlib(real_np)
            

    #         # if len(boxes):
    #         #     # pick largest face
    #         #     x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    #         #     real_pil = real_pil.crop((x1, y1, x2, y2))
    #         #     fake_pil = fake_pil.crop((x1, y1, x2, y2))  # same crop
        

    #         # ------------------- TENSOR CONVERSION ------------------- #
    #         real_t = TF.to_tensor(real_pil).unsqueeze(0)   # [1,C,H,W]
    #         fake_t = TF.to_tensor(fake_pil).unsqueeze(0)

    #         # augment separately (but same resize first)
    #         real = self.real_aug(real_t).squeeze(0)
    #         fake = self.fake_aug(fake_t).squeeze(0)

    #         # force size → 224x224
    #         real_t = F.interpolate(real_t, size=(224, 224), mode="bilinear", align_corners=False)[0]
    #         fake_t = F.interpolate(fake_t, size=(224, 224), mode="bilinear", align_corners=False)[0]
        
    #         # diff image (normalized)
    #         diff_image = (real_t - fake_t).abs().float()
    #         IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
    #         IMAGENET_STD  =  [0.26862954, 0.26130258, 0.27577711]
           
    #         # IMAGENET_MEAN = [0.485, 0.456, 0.406]
    #         # IMAGENET_STD = [0.229, 0.224, 0.225]
    #         # after resizing / augmentations
    #         real_t = TF.normalize(real_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    #         fake_t = TF.normalize(fake_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    #         return real_t, fake_t, diff_image, torch.tensor(label, dtype=torch.long)

    #     except Exception:
    #         traceback.print_exc()
    #         return self.__getitem__((index + 1) % len(self.file_list))
