import os
import random
import traceback
from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from audiomentations import (
    AddGaussianNoise,
    BandPassFilter,
    Gain,
    Normalize,
    PitchShift,
    PolarityInversion,
    SevenBandParametricEQ,
    Shift,
    TimeStretch,
)

from skimage import transform as trans
from torchvision.transforms import functional as TF
from decord import VideoReader, cpu
from kornia import augmentation as K
from torch.utils.data import Dataset
from transformers import BertTokenizer

import torch.nn.functional as F
from PIL import Image
import cv2
# import dlib
import insightface

# import dlib
import insightface

# Alternative face detectors - choose one
try:
    import dlib
    DLIB_AVAILABLE = True
    dlib_detector = dlib.get_frontal_face_detector()
    SHAPE_PREDICTOR = "<path_to_dataset>/shape_predictor_68_face_landmarks.dat"
    dlib_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)  # Download from dlib
except ImportError:
    DLIB_AVAILABLE = False




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

# OpenCV Haar Cascade (always available with OpenCV)
try:
    haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    HAAR_AVAILABLE = True
except:
    HAAR_AVAILABLE = False

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

import mediapipe as mp

# Initialize MediaPipe face detector once
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4)

def preprocess_image(image, size: int = 224, equalize: bool = True) -> torch.Tensor:
    """
    Load image -> detect face using MediaPipe -> crop -> equalize (optional) -> resize -> normalize (CLIP mean/std).
    Returns tensor [3,H,W].
    """
    bgr = image
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape

    # Run MediaPipe face detection
    results = face_detector.process(rgb)
    
    # --- Select best detection if available ---
    if results.detections:
        # pick the largest face (by area)
        boxes = []
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x0 = int(bbox.xmin * w)
            y0 = int(bbox.ymin * h)
            x1 = int((bbox.xmin + bbox.width) * w)
            y1 = int((bbox.ymin + bbox.height) * h)
            boxes.append((x0, y0, x1, y1))
        # pick largest bounding box
        x0, y0, x1, y1 = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))

        # Slightly enlarge to include jawline
        pad_w = int(0.01 * (x1 - x0))
        pad_h = int(0.01 * (y1 - y0))
        x0 = max(0, x0 - pad_w)
        y0 = max(0, y0 - pad_h)
        x1 = min(w, x1 + pad_w)
        y1 = min(h, y1 + pad_h)

        face_rgb = rgb[y0:y1, x0:x1]
    else:
        # fallback to full frame
        face_rgb = rgb

    # Optional Y-channel equalization for lighting normalization
  
    # Resize for CLIP input
    face_rgb = cv2.resize(face_rgb, (size, size), interpolation=cv2.INTER_AREA)
    return face_rgb
    # # Convert to tensor and normalize
    # arr = face_rgb.astype(np.float32) / 255.0
    # chw = torch.from_numpy(arr).permute(2, 0, 1)
    # chw = TF.normalize(chw, mean=CLIP_MEAN, std=CLIP_STD)
    # return chw



def safe_face_crop(img_rgb, out_size=(224, 224), enforce_square=True):
    """
    Detect face and perform a contextual crop optimized for deepfake detection.
    Increased border/context is added, and optionally enforces a square crop aspect ratio.
    If no face detected, returns a central crop.
    """
    if img_rgb is None or img_rgb.size == 0:
        raise ValueError("safe_face_crop(): invalid or empty image input")

    h, w = img_rgb.shape[:2]
    
    # Use BGR input for the InsightFace detector
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # The detector is not defined here, assume it's available globally or passed in
    # For a robust solution, you might pass the detector as an argument
    try:
        faces = insight_detector.get(bgr) 
    except NameError:
        print("Warning: insight_detector not found. Falling back to center crop.")
        faces = []

    x1_crop, y1_crop, x2_crop, y2_crop = 0, 0, w, h

    # ---------------------------------------------------
    # 1. FACE DETECTED → Calculate expanded bounding box
    # ---------------------------------------------------
    if faces and len(faces) > 0:
        # Pick the largest detected face
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        x1_det, y1_det, x2_det, y2_det = f.bbox.astype(int)
        
        det_w = x2_det - x1_det
        det_h = y2_det - y1_det
        
        # --- Deepfake Detection Expansion Factors (Increased Context) ---
        # Increased context, especially on the top (hairline) and bottom (neck/collar),
        # where blending artifacts and inconsistencies are common.
        # EXP_LEFT, EXP_TOP, EXP_RIGHT, EXP_BOTTOM = 0.35, 0.45, 0.35, 0.35 # Example aggressive factors
        
        # Use more balanced, but still generous factors
        EXP_LEFT, EXP_TOP, EXP_RIGHT, EXP_BOTTOM = 0.01, 0.01, 0.01, 0.01

        # Base expansion size, often based on the maximum dimension
        # Base expansion size, often based on the maximum dimension
        base_exp = max(det_w, det_h) 

        # Calculate expansion based on base_exp for more uniform context
        x_exp_l = int(base_exp * EXP_LEFT)
        y_exp_t = int(base_exp * EXP_TOP)
        x_exp_r = int(base_exp * EXP_RIGHT)
        y_exp_b = int(base_exp * EXP_BOTTOM)

        # Apply expansion
        x1_crop_raw = x1_det - x_exp_l
        y1_crop_raw = y1_det - y_exp_t
        x2_crop_raw = x2_det + x_exp_r
        y2_crop_raw = y2_det + y_exp_b
        # Enforce a Square Aspect Ratio (Highly recommended for CNN detectors)
        if enforce_square:
            crop_w_raw = x2_crop_raw - x1_crop_raw
            crop_h_raw = y2_crop_raw - y1_crop_raw
            
            # Determine the side of the square (use the larger dimension)
            square_side = max(crop_w_raw, crop_h_raw)
            
            # Center the crop box
            cx = (x1_crop_raw + x2_crop_raw) // 2
            cy = (y1_crop_raw + y2_crop_raw) // 2
            
            # Recalculate crop coordinates for a square box
            x1_crop_sq = cx - square_side // 2
            y1_crop_sq = cy - square_side // 2
            x2_crop_sq = cx + (square_side - square_side // 2) # Handles odd 'square_side'
            y2_crop_sq = cy + (square_side - square_side // 2)

            x1_crop_raw, y1_crop_raw = x1_crop_sq, y1_crop_sq
            x2_crop_raw, y2_crop_raw = x2_crop_sq, y2_crop_sq
            
        # Clamp to image boundaries
        x1_crop = max(0, x1_crop_raw)
        y1_crop = max(0, y1_crop_raw)
        x2_crop = min(w, x2_crop_raw)
        y2_crop = min(h, y2_crop_raw)

        # Final validity check (optional: could fallback to unexpanded detection box)
        if x2_crop <= x1_crop or y2_crop <= y1_crop:
            x1_crop, y1_crop, x2_crop, y2_crop = x1_det, y1_det, x2_det, y2_det

    # ---------------------------------------------------
    # 2. NO FACE DETECTED or INVALID BOX → Fallback center crop
    # ---------------------------------------------------
    else: # This path is taken if 'faces' list is empty or invalid
        # Use a square side that is a large fraction of the minimum image dimension
        # to capture potential forgery even without a detected face (e.g., body forgery)
        side = min(h, w)
        cx, cy = w // 2, h // 2
        
        # Use a larger central crop, e.g., 80% of the minimum side
        half_side = int(side * 0.4) 
        
        x1_crop = max(0, cx - half_side)
        y1_crop = max(0, cy - half_side)
        x2_crop = min(w, cx + half_side)
        y2_crop = min(h, cy + half_side)

    # ---------------------------------------------------
    # 3. PERFORM CROP AND RESIZE
    # ---------------------------------------------------
    
    # Perform the crop using NumPy slicing
    cropped = img_rgb[y1_crop:y2_crop, x1_crop:x2_crop]

    if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
        # Final absolute fallback (e.g., if image is tiny or crop indices were strange)
        # Use a smaller, safe central region.
        cropped = img_rgb[h//4:3*h//4, w//4:3*w//4]

    # Resize to the final output size
    # Using INTER_LINEAR is sometimes preferred over INTER_AREA for detail preservation, 
    # but INTER_AREA is good for image shrinking (downsampling).
    return cv2.resize(cropped, out_size, interpolation=cv2.INTER_LINEAR)


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
###########################################################################
# ----------------------------  helper utils  --------------------------- #
###########################################################################

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

    # PIL.Images
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


# Make Yunet model path configurable
YUNET_MODEL = os.environ.get("YUNET_MODEL", "<path_to_dataset>/yunet_2022mar.onnx")
yunet = cv2.FaceDetectorYN.create(
    model=YUNET_MODEL,
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

def to_pil(tensor):
    # Handle [B,C,H,W] or [C,H,W]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)   # drop batch
    if tensor.dtype != torch.uint8:
        tensor = (tensor.clamp(0,1) * 255).to(torch.uint8)
    return TF.to_pil_image(tensor)

def equalize_frames(frames_np: np.ndarray) -> torch.Tensor:
    """Histogram‑equalise luminance channel of a (T, H, W, C) RGB numpy array."""
    eq = []
    for f in frames_np:
        ycrcb = cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)
        ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
        eq.append(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB))
    eq = np.stack(eq, 0)  # (T, H, W, C)
    return torch.from_numpy(eq).permute(0, 3, 1, 2).float() / 255  # (T, C, H, W)


def get_image_path(root, method, pair_filename, compression="raw"):

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

def get_real_path(root, video_id,compression):
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

###########################################################################
# --------------------------  MAIN DATASET  ----------------------------- #
###########################################################################

class FaceswapImagesDataset(Dataset):
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

        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.frame_rate = frame_rate  # Number of frames to extract per second
        self.real_dir = os.path.join(base_dir, f"{partition}/real")
        print(f"{self.base_dir}deepspeak/dataset/")
        # self.face_detector = fasterrcnn_resnet50_fpn(pretrained=True)
        # self.face_detector.eval()  # Set model to evaluation mode
        self.fake_dir = os.path.join(base_dir, f"{partition}/fake")
        video_files = {}
        if partition=="train":
            self.is_train=True
        else:
            self.is_train=False
        # ---------------------------------------------------------------
        # AUDIO pipeline  
        # ---------------------------------------------------------------
      

        self.real_aug = K.AugmentationSequential(
            
            K.Resize((224, 224)),
            # K.RandomHorizontalFlip(p=0.9),
            # K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.9),
            # K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.9),
            # K.RandomMotionBlur(kernel_size=3, angle=15., direction=0.5,p=0.9),
            # K.RandomAffine(degrees=4,
            #                translate=(0.02, 0.02),
            #                scale=(0.9, 1.1),
            #                shear=None,
            #                p=0.6),
            # K.RandomErasing(scale=(0.02, 0.33), ratio=(0.5, 3.6), p=0.9),      # sim. occlusion
            # same_on_batch=True  # <- this is key
            # data_format="BCTHW"
        )
        self.fake_aug = K.AugmentationSequential(
            
            K.Resize((224, 224)),
            # K.RandomHorizontalFlip(p=0.9),
            # K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.9),
            # K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.9),
            # K.RandomMotionBlur(kernel_size=3, angle=15., direction=0.5,p=0.9),
            # K.RandomAffine(degrees=4,
            #                translate=(0.02, 0.02),
            #                scale=(0.9, 1.1),
            #                shear=None,
            #                p=0.6),
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

                
        print("started to load data")
        
              
        self.file_list = []
        ff_methods = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
        external_datasets = ["CelebDF", "WildFake", "DFDC", "simswap", "blendface","34s","facedancer","fsgan","inswap","mobileswap","uniface"]
        # Initialize fake category lists
        fake_methods = {
            "Deepfakes": [],
            "Face2Face": [],
            "FaceShifter": [],
            "FaceSwap": [],
            "NeuralTextures": []
        }
        real_paths = []
        MAX_FRAMES = 6
        FAKE = 0
        REAL = 0
        cache_file = f"faceswap_ds_{partition}_{take_datasets[0]}.json"
        
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
                # self.fake_image_list.append(item["fake_path"])   # real image path
                self.label_list.append(item["label"])  # label (0 or 1 etc.)

            self.data_dict = {
                # 'image': self.fake_image_list,
                'image_real': self.real_image_list,
                'label': self.label_list
            }
            print(f"Loaded {len(self.file_list)} files from cache.")
            return
        external_dataset = any(ds in external_datasets for ds in take_datasets)
        # ────────────────────────────────────────────────────────────
        for compression in ["c23"]:
            if external_dataset:
                continue
            ff_root = Path(f"<path_to_dataset>/faceforensics++/face_data/data_{compression}")
            
            if partition == "train":
                df = pd.read_json(os.path.join(ff_root,"train.json"))
            else:
                df = pd.read_json(os.path.join(ff_root,"val.json"))

            # for _, row in df.iterrows():
            #     pair1 = str(row[0]).zfill(3)
            #     pair2 = str(row[1]).zfill(3)
            #     pair_ids = [f"{pair1}_{pair2}.avi", f"{pair2}_{pair1}.avi"]

                # Check for both directions of fake videos
                # for pair_filename in pair_ids:
                #     for method, path_list in fake_methods.items():
                #         fake_path = get_image_path(root, method, pair_filename)
                #         # check_and_add(fake_path, path_list)
                #         if os.path.exists(fake_path):
                #             self.file_list.append({"path":fake_path,"label":0})

                # Check real videos for both video IDs
            
            fake_methods = {}
            for data in take_datasets: 
                fake_methods[data] = []
                
            # fake_methods = {"NeuralTextures": []}
            for _, row in df.iterrows():
                p1, p2 = str(row[0]).zfill(3), str(row[1]).zfill(3)
                pair_ids = [f"{p1}_{p2}", f"{p2}_{p1}"]
                # fake
                for pair in pair_ids:
                    for method in fake_methods:
                        fp = get_image_path(ff_root, method, pair, compression)
                        if os.path.exists(fp):
                            frames = 0 
                            for file in os.listdir(fp):
                                if file.endswith(".jpg") or file.endswith(".png"):
                                    fp_path = os.path.join(fp, file)
                                    if os.path.exists(fp_path):
                                        
                                        if frames>MAX_FRAMES:
                                            break
                                        frames+=1
                                        FAKE+=1
                                        self.file_list.append({"path": fp_path, "label": 1, "multi_label":method})
                # real
                for vid in (p1, p2):
                    rp = get_real_path(ff_root, vid, compression)
                    if os.path.exists(rp):
                        frames = 0 
                        for file in os.listdir(rp):
                            if file.endswith(".jpg") or file.endswith(".png"):
                                
                                rp_path = os.path.join(rp, file)
                                if os.path.exists(rp_path):
                                    
                                    if frames>MAX_FRAMES:
                                        break
                                    frames+=1
                                    REAL+=1
                                    self.file_list.append({"path": rp_path, "label": 0, "multi_label":"real"})

    

        if "DFD_simple" in take_datasets:
            celeb_root = Path("<path_to_dataset>/DFD/cropped_samples")
            for type in ["real","fake"]:
                type_root = os.path.join(celeb_root, type)
                for folder in os.listdir(type_root):
                    folder_path = os.path.join(type_root, folder)
                    label = 0 if type=="real" else 1
                    frames = 0 
                    for file in os.listdir(folder_path):
                        
                        if frames>MAX_FRAMES:
                            break
                        frames+=1
                        file_path = os.path.join(folder_path, file)
                        if os.path.exists(file_path):
                            if label == 0:
                                REAL+=1
                            else:
                                FAKE+=1
                            self.file_list.append({"path": file_path, "label": label, "multi_label":"real"})
            
        df40_ds= ["simswap", "blendface","34s","facedancer","fsgan","inswap","mobileswap","uniface"]
        for df40 in df40_ds:
            if df40 in take_datasets:
                wf_root = Path(f"<path_to_dataset>/DF40/preprocessed/test/data/{df40}/ff/frames/")
                image_exts = (".jpg", ".jpeg", ".png") 
                    # Walk through all subdirectories
                for root, _, files in os.walk(wf_root):
                    frames = 0
                    for file in files:
                        if file.lower().endswith(image_exts):
                            file_path = os.path.join(root, file)
                            if os.path.exists(file_path):
                                
                                if frames>MAX_FRAMES:
                                    break
                                FAKE += 1
                                frames+=1
                                self.file_list.append({
                                    "path": file_path,
                                    "label": 1,
                                    "multi_label": "fake"
                                })

                ff_root = Path(f"<path_to_dataset>/faceforensics++/face_data/data_raw")
                df = pd.read_json(os.path.join(ff_root,"test.json"))
                fake_methods = {}
                for data in take_datasets: 
                    fake_methods[data] = []
                    
                # fake_methods = {"NeuralTextures": []}
                for _, row in df.iterrows():
                    p1, p2 = str(row[0]).zfill(3), str(row[1]).zfill(3)
                    pair_ids = [f"{p1}_{p2}", f"{p2}_{p1}"]
                
                    # real
                    for vid in (p1, p2):
                        rp = get_real_path(ff_root, vid, "raw")
                        if os.path.exists(rp):
                            frames = 0 
                            for file in os.listdir(rp):
                                if file.endswith(".jpg") or file.endswith(".png"):
                                    
                                    rp_path = os.path.join(rp, file)
                                    if os.path.exists(rp_path):
                                        
                                        if frames>MAX_FRAMES:
                                            break
                                        frames+=1
                                        REAL+=1
                                        self.file_list.append({"path": rp_path, "label": 0, "multi_label":"real"})

        if "CelebDFV2" in take_datasets:
            wf_root = Path("<path_to_dataset>/dataset/Celeb-DF-v2/Celeb-synthesis/frames")
          
            image_exts = (".jpg", ".jpeg", ".png") 
                # Walk through all subdirectories
            for root, _, files in os.walk(wf_root):
                frames = 0 
                for file in files:
                    if file.lower().endswith(image_exts):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            
                            if frames>MAX_FRAMES:
                                break
                            FAKE += 1
                            frames+=1
                            self.file_list.append({
                                "path": file_path,
                                "label": 1,
                                "multi_label": "fake"
                            })

                            
            wf_root = Path("<path_to_dataset>/dataset/Celeb-DF-v2/Celeb-real/frames")
          
            image_exts = (".jpg", ".jpeg", ".png") 
                # Walk through all subdirectories
            for root, _, files in os.walk(wf_root):
                frames = 0 
                for file in files:
                    if file.lower().endswith(image_exts):
                       
                        if frames>MAX_FRAMES:
                            break
                        frames+=1
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            REAL += 1
                            self.file_list.append({
                                "path": file_path,
                                "label": 0,
                                "multi_label": "real" 
                            })

                            
            wf_root = Path("<path_to_dataset>/dataset/Celeb-DF-v2/YouTube-real/frames")
          
            image_exts = (".jpg", ".jpeg", ".png") 
                # Walk through all subdirectories
            for root, _, files in os.walk(wf_root):
                frames = 0 
                for file in files:
                    if file.lower().endswith(image_exts):
                       
                        if frames>MAX_FRAMES:
                            break
                        frames+=1
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            REAL += 1
                            self.file_list.append({
                                "path": file_path,
                                "label": 0,
                                "multi_label": "real" 
                            })

                        
        if "DFDCp" in take_datasets:
            wf_root = Path("<path_to_dataset>/DFDC-Official/test_dfdc_preview_face_set/method_A")
          
            image_exts = (".jpg", ".jpeg", ".png") 
                # Walk through all subdirectories
            for root, _, files in os.walk(wf_root):
                frames = 0 
                for file in files:
                    if file.lower().endswith(image_exts):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            
                            if frames>MAX_FRAMES:
                                break
                            FAKE += 1
                            frames+=1
                            self.file_list.append({
                                "path": file_path,
                                "label": 1,
                                "multi_label": "fake"
                            })
            wf_root = Path("<path_to_dataset>/DFDC-Official/test_dfdc_preview_face_set/method_B")
          
            image_exts = (".jpg", ".jpeg", ".png") 
                # Walk through all subdirectories
            for root, _, files in os.walk(wf_root):
                frames = 0 
                for file in files:
                    if file.lower().endswith(image_exts):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            
                            if frames>MAX_FRAMES:
                                break
                            FAKE += 1
                            frames+=1
                            self.file_list.append({
                                "path": file_path,
                                "label": 1,
                                "multi_label": "fake"
                            })

                            
            wf_root = Path("<path_to_dataset>/DFDC-Official/test_dfdc_preview_face_set/original_videos")
          
            image_exts = (".jpg", ".jpeg", ".png") 
                # Walk through all subdirectories
            for root, _, files in os.walk(wf_root):
                frames = 0
                for file in files:
                    if file.lower().endswith(image_exts):
                       
                        if frames>MAX_FRAMES:
                            break
                        frames+=1
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            REAL += 1
                            self.file_list.append({
                                "path": file_path,
                                "label": 0,
                                "multi_label": "real" 
                            })

                        



        if "FFIW" in take_datasets:
            wf_root = Path("<path_to_dataset>/FFIW/FFIW10K-v1-release/cropped_video/target/val")
          
            image_exts = (".jpg", ".jpeg", ".png") 
                # Walk through all subdirectories
            for root, _, files in os.walk(wf_root):
                frames = 0 
                for file in files:
                    if file.lower().endswith(image_exts):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            
                            if frames>MAX_FRAMES:
                                break
                            FAKE += 1
                            frames+=1
                            self.file_list.append({
                                "path": file_path,
                                "label": 1,
                                "multi_label": "fake"
                            })

                            
            wf_root = Path("<path_to_dataset>/FFIW/FFIW10K-v1-release/cropped_video/source/val")
          
            image_exts = (".jpg", ".jpeg", ".png") 
                # Walk through all subdirectories
            for root, _, files in os.walk(wf_root):
                frames = 0 
                for file in files:
                    if file.lower().endswith(image_exts):
                       
                        if frames>MAX_FRAMES:
                            break
                        frames+=1
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            REAL += 1
                            self.file_list.append({
                                "path": file_path,
                                "label": 0,
                                "multi_label": "real" 
                            })

        if "DF0" in take_datasets:
            wf_root = Path("<path_to_dataset>/DeeperForensics1.0/cropping/fake")
          
            image_exts = (".jpg", ".jpeg", ".png") 
                # Walk through all subdirectories
            for root, _, files in os.walk(wf_root):
                frames = 0 
                for file in files:
                    if file.lower().endswith(image_exts):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            
                            if frames>MAX_FRAMES:
                                break
                            FAKE += 1
                            frames+=1
                            self.file_list.append({
                                "path": file_path,
                                "label": 1,
                                "multi_label": "fake"
                            })

                            
            wf_root = Path("<path_to_dataset>/DeeperForensics1.0/cropping/real")
          
            image_exts = (".jpg", ".jpeg", ".png") 
                # Walk through all subdirectories
            for root, _, files in os.walk(wf_root):
                frames = 0 
                for file in files:
                    if file.lower().endswith(image_exts):
                       
                        if frames>MAX_FRAMES:
                            break
                        frames+=1
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            REAL += 1
                            self.file_list.append({
                                "path": file_path,
                                "label": 0,
                                "multi_label": "real" 
                            })

        if "WildFake" in take_datasets:
            wf_root = Path("<path_to_dataset>/WildDeepFake/original/fake_test/imgs")
          
            image_exts = (".jpg", ".jpeg", ".png") 
                # Walk through all subdirectories
            for root, _, files in os.walk(wf_root):
                frames = 0 
                for file in files:
                    if file.lower().endswith(image_exts):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            
                            if frames>MAX_FRAMES:
                                break
                            FAKE += 1
                            frames+=1
                            self.file_list.append({
                                "path": file_path,
                                "label": 1,
                                "multi_label": "fake"
                            })

                            
            wf_root = Path("<path_to_dataset>WildDeepFake/original/real_test/imgs")
          
            image_exts = (".jpg", ".jpeg", ".png") 
                # Walk through all subdirectories
            for root, _, files in os.walk(wf_root):
                frames = 0 
                for file in files:
                    if file.lower().endswith(image_exts):
                       
                        if frames>MAX_FRAMES:
                            break
                        frames+=1
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            REAL += 1
                            self.file_list.append({
                                "path": file_path,
                                "label": 0,
                                "multi_label": "real" 
                            })

       
        if "DFDC" in take_datasets:
            wf_root = Path("<path_to_dataset>/DFDC-Official/full_set/val/frames/real")
                # Walk through all subdirectories
            for directory in os.listdir(wf_root):
                frames=0 
                dir_path = os.path.join(wf_root, directory)
                for file in os.listdir(dir_path):
                  
                    if frames>MAX_FRAMES:
                        break
                    frames+=1
                    file_path = os.path.join(dir_path, file)
                    if os.path.exists(file_path):
                        REAL += 1
                        self.file_list.append({
                            "path": file_path,
                            "label": 0,
                            "multi_label": "real"
                        })

            wf_root = Path("<path_to_dataset>/DFDC-Official/full_set/val/frames/fake")
                # Walk through all subdirectories
            for directory in os.listdir(wf_root):
                frames=0 
                dir_path = os.path.join(wf_root, directory)
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.exists(file_path):
                      
                        if frames>MAX_FRAMES:
                            break
                        FAKE += 1
                        frames+=1
                        self.file_list.append({
                            "path": file_path,
                            "label": 1,
                            "multi_label": "fake"
                        })


        # Load JSON file
        if "DFDC_test" in take_datasets:
            df = pd.read_json(
                "<path_to_dataset>/dataset/DFDC/test/metadata.json"
            ).T.reset_index().rename(columns={"index": "filename"})

            # Now you have filename + is_fake (+ augmentations if needed)
            # print(df.head())


            folder = "<path_to_dataset>/dataset/DFDC/test"

            for _, row in df.iterrows():
                filename = row["filename"]      # "aalscayrfi.mp4"
                label = row["is_fake"]          # 0 or 1

                video_folder = os.path.join(folder, "frames", filename.replace(".mp4", ""))
                if not os.path.exists(video_folder):
                    continue

                frames = 0
                for file in os.listdir(video_folder):
                    if frames > MAX_FRAMES:
                        break
                    file_path = os.path.join(video_folder, file)
                    if os.path.exists(file_path):
                        frames += 1
                        if label == 0:
                            REAL += 1
                        else:
                            FAKE += 1
                        self.file_list.append({
                            "path": file_path,
                            "label": label,
                            "multi_label": "real" if label == 0 else "fake"
                        })

        # Save built file_list to cache
        with open(cache_file, "w") as f:
            json.dump(self.file_list, f, indent=2)
        print(f"Total files in {partition}: {len(self.file_list)}, Total Real: {REAL}, Total Fake: {FAKE}")


        
        print(f"Total files in {partition} are {len(self.file_list)} {take_datasets} TOTAL FAKES: {FAKE}, TOTAL REALS: {REAL}")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):            
        while True:  # loop until a valid sample is returned
            try:
                file_path = self.file_list[index]["path"]
                label = self.file_list[index]["label"]

                # Load image as PIL
                image_pil = Image.open(file_path).convert("RGB")
                image_np = np.array(image_pil)  # RGB
                # image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                # boxes = detect_faces_insight(image_bgr)

                # if len(boxes) > 0:
                #     # pick largest face
                #     x1, y1, x2, y2, _ = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                #     image_pil = image_pil.crop((x1, y1, x2, y2))
                # Face detection
                # boxes = _detect_yunet_boxes_bgr(image_bgr)
                # boxes = detect_faces_dlib(image)
                            
                # if len(boxes) > 0:
                #     # pick largest face
                #     # best = max(boxes, key=lambda b: (b.right()-b.left())*(b.bottom()-b.top()))
                #     # x1, y1, x2, y2 = best.left(), best.top(), best.right(), best.bottom()

                #     x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                #     image_pil = image_pil.crop((x1, y1, x2, y2))
            
                # Convert PIL to torch tensor [C,H,W] in [0,1]
                # image = TF.to_tensor(image_pil).unsqueeze(0)  # add batch

                image = np.array(image_np)
                # --- Detect landmarks on REAL first (fallback to FAKE) ---
            #     bbox, landmarks_2d = detect_faces_with_landmarks(image, method="dlib")
                    
            #     if landmarks_2d is None or bbox is None:
            #         bbox, landmarks_2d = detect_faces_with_landmarks(image, method="dlib")
                
            # # --- Build a 5-point set for alignment ---
            #     def bbox_to_5pt(b):
            #         if b is None:
            #             h, w = image.shape[:2]
            #             x, y, bw, bh = int(0.1*w), int(0.1*h), int(0.8*w), int(0.8*h)
            #         else:
            #             x, y, bw, bh = b
            #         le = (x + 0.35 * bw, y + 0.40 * bh)
            #         re = (x + 0.65 * bw, y + 0.40 * bh)
            #         no = (x + 0.50 * bw, y + 0.55 * bh)
            #         ml = (x + 0.38 * bw, y + 0.75 * bh)
            #         mr = (x + 0.62 * bw, y + 0.75 * bh)
            #         return np.array([le, re, no, ml, mr], dtype=np.float32)

            #     if landmarks_2d is not None and len(landmarks_2d) > 0:
            #         n = landmarks_2d.shape[0]
            #         if n == 68:
            #             lm5 = landmarks68_to_5(landmarks_2d)
            #         elif n == 5:
            #             lm5 = landmarks_2d.astype(np.float32)
            #         else:
            #             lm5 = bbox_to_5pt(bbox)
            #     else:
            #         lm5 = bbox_to_5pt(bbox)
                real_aligned = image
                if "DFDCp" in self.data_labels or "FIFW" in self.data_labels or "DF0" in self.data_labels or "DFDC_val" in self.data_labels:
                    real_aligned = safe_face_crop(image)
            
                # Apply augmentations
                image = self.real_aug(real_aligned)  # Kornia expects BCHW
                image_aug = self.real_aug_ori(image)

                # final resize (ensures fixed size for model)
                image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
                image_aug = F.interpolate(image_aug, size=(224, 224), mode="bilinear", align_corners=False)
                
                IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
                IMAGENET_STD  =  [0.26862954, 0.26130258, 0.27577711]
           
        

                image = TF.normalize(image.squeeze(0), mean=IMAGENET_MEAN, std=IMAGENET_STD)
                image_aug = TF.normalize(image_aug.squeeze(0), mean=IMAGENET_MEAN, std=IMAGENET_STD)
                return image.squeeze(0), image_aug.squeeze(0), torch.tensor(label, dtype=torch.long)

            except Exception as e:
                traceback.print_exc()
                print(f"Error loading {file_path}: {e}")
                index = (index + 1) % len(self.file_list)
