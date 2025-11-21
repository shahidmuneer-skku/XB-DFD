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
from PIL import Image
from decord import VideoReader
from decord import cpu, gpu
import torchvision.transforms as T
import subprocess  # Missing import for extract_audio_with_ffmpeg
# from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Normalize, PolarityInversion
from transformers import BertTokenizer
import pandas as pd
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from decord import VideoReader
from decord import cpu, gpu
import librosa
from pathlib import Path
from audiomentations import Compose,SevenBandParametricEQ, AddGaussianNoise, TimeStretch, PitchShift, Shift, Normalize, PolarityInversion, Gain, BandPassFilter, Reverse, Clip
import torch
import kornia.augmentation as K
import kornia.geometry as KG
# from vidaug import augmentors as va
import torchvision.transforms.functional as TF
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

def pad_random(x: np.ndarray, max_len: int = 64000):
    x_len = x.shape[0]
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))
    return pad_random(padded_x, max_len)

def _pair_real_fake(real_root: Path,
                    fake_root: Path,
                    quality_tag: str,
                    file_list: list):
    """
    Add (real_path, fake_path) pairs for one quality level.

    Args
    ----
    real_root   Path :  directory tree with REAL videos  (e.g. .../data_raw/.../original_sequences)
    fake_root   Path :  directory tree with FAKE videos  (e.g. .../data_raw/.../manipulated_sequences)
    quality_tag str  :  "raw" or "c40"  (only used for debug / warnings)
    file_list   list :  list that will be extended in-place
    """
    # Real folders: key = '113', val = path
    real_dirs = {d.name: d for d in real_root.iterdir() if d.is_dir()}

    # Fake folders: extract second half (e.g., '983_113' → '113')
    fake_dirs = {}
    for d in fake_root.iterdir():
        if d.is_dir():
            parts = d.name.split("_")
            if len(parts) == 2:
                fake_dirs[parts[1]] = d

    common = sorted(real_dirs.keys() & fake_dirs.keys())
   
    if not common:
        raise RuntimeError(f"[{quality_tag}] No common real/fake folders matched.")

    for vid in common:
        r_dir = real_dirs[vid]
        f_dir = fake_dirs[vid]

        real_imgs = sorted([p for p in r_dir.rglob("*.png") if p.is_file()])
        fake_imgs = sorted([p for p in f_dir.rglob("*.png") if p.is_file()])

        if len(real_imgs) != len(fake_imgs):
            print(f"[WARN:{quality_tag}] {vid}: #real={len(real_imgs)} ≠ #fake={len(fake_imgs)}")

        for real_f, fake_f in zip(real_imgs, fake_imgs):
            file_list.append({
                "path":      str(real_f),   # real image
                "fake_path": str(fake_f),   # fake image
                "label":     0
            })

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


import torch
import librosa
import cv2
import numpy as np
from decord import VideoReader, cpu

def read_video(path: str):
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
        num_frames = len(vr)  # total frames in the video
        # shape: (T, H, W, C) in RGB order
        frames_array = vr.get_batch(range(num_frames)).asnumpy()  
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
    audio_path = "<path_to_dataset>/cropped_video.wav"
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
    except Exception as e:
        print(f"Error reading audio from {audio_path} with librosa: {e}")
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

def normalize_landmarks_standardize(landmarks):
 
    mean = np.mean(landmarks, axis=0)
    std = np.std(landmarks, axis=0)
    std[std == 0] = 1

    return (landmarks - mean) / std

class FaceswapDataset(Dataset):
    """
    Dataset class for the SAMMD2024 dataset.
    """
    def __init__(self, base_dir, partition="train",take_datasets="1,2,3,4,5,6", max_len=64000, frame_rate=1, n_mfcc=40):
        assert partition in ["train", "dev", "test"], "Invalid partition. Must be one of ['train', 'dev', 'test']"
        self.base_dir = base_dir
        self.data_labels = take_datasets
        self.partition = partition
        self.max_len = max_len
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.frame_rate = frame_rate  # Number of frames to extract per second
        self.real_dir = os.path.join(base_dir, f"{partition}/real")
        self.face_detector = fasterrcnn_resnet50_fpn(pretrained=True)
        self.face_detector.eval()  # Set model to evaluation mode
        self.fake_dir = os.path.join(base_dir, f"{partition}/fake")
        video_files = {}
        if partition=="train":
            self.is_train=True
        else:
            self.is_train=False
       

        self.image_aug = K.AugmentationSequential(
            # Resize to match model input
            K.Resize((224, 224)),

            # Spatial transformations
            K.RandomHorizontalFlip(p=0.95),
            K.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.85, 1.15),
                shear=10.0,
                p=0.7
            ),
            K.RandomPerspective(distortion_scale=0.3, p=0.5),
            K.RandomElasticTransform(alpha=(0.9, 1.3), sigma=(7.0, 13.0), p=0.8),

            # Appearance & color distortion
            K.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                p=0.9
            ),
            K.RandomGrayscale(p=0.2),
            K.RandomGamma(gamma=(0.7, 1.3), p=0.6),

            # Blurs and motion simulation
            K.RandomGaussianBlur((5, 5), (0.2, 2.5), p=0.9),
            K.RandomMotionBlur(kernel_size=5, angle=20., direction=0.7, p=0.9),

            # Occlusions / corruptions
            K.RandomErasing(scale=(0.02, 0.4), ratio=(0.4, 3.6), p=0.95),

            # Compression and artifacts
            K.RandomJPEG(jpeg_quality=(20.0, 60.0), p=0.85),
            K.RandomBoxBlur(kernel_size=(3, 3), p=0.5),

            # Noise injection
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.8),

            # Final settings
            data_keys=["input"],
            same_on_batch=False,
        )


        # self.image_aug = K.AugmentationSequential(
        #     K.Resize((224, 224)),
        #     K.RandomHorizontalFlip(p=0.9),
        #     K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.9),
        #     K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.9),
        #     K.RandomMotionBlur(kernel_size=3, angle=15., direction=0.5,p=0.9),
        #     K.RandomAffine(degrees=4,
        #                    translate=(0.02, 0.02),
        #                    scale=(0.9, 1.1),
        #                    shear=None,
        #                    p=0.6),
        #     K.RandomErasing(scale=(0.02, 0.33), ratio=(0.5, 3.6), p=0.9),      # sim. occlusion
        #     # data_format="BCTHW"
        # )



        self.image_aug_ori = K.AugmentationSequential(
            # Resize to match input shape
            K.Resize((224, 224)),

            # Spatial augmentations
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), p=0.3),
            K.RandomRotation(degrees=5.0, p=0.2),

            # Color & intensity jitter
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            K.RandomGamma(gamma=(0.8, 1.2), p=0.2),
            K.RandomGrayscale(p=0.1),

            # Blur and noise augmentations
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.4),
            K.RandomMotionBlur(kernel_size=3, angle=15.0, direction=0.5, p=0.3),
            K.RandomPerspective(distortion_scale=0.2, p=0.2),

            # Occlusion-style robustness
            K.RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 3.3), same_on_batch=False, p=0.3),
            K.RandomBoxBlur(kernel_size=(5, 5), p=0.2),
            
            # Noise injection
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.3),

            K.RandomJPEG(jpeg_quality=(20.0, 60.0), p=0.2),
            # Optional: JPEG compression artifacts
            # K.RandomJPEGCompression(quality=(70, 100), p=0.2),  # realistic compression artifacts

            # Keep format consistent
            data_keys=["input"]
        )

              
        self.image_no_aug = K.AugmentationSequential(
                    K.Resize((224, 224)),
                    # data_format="BCTHW"
                )
        self.file_list = []
        
                     
        if partition=="test":
          

            
            root_dir = "<path_to_dataset>/data_c40/original_sequences/youtube/c40/v1faces"
                    
            for directory in os.listdir(root_dir):
                dir_path = os.path.join(root_dir, directory)

                for file in os.listdir(dir_path):
                    full_path = os.path.join(dir_path, file)
                    # print(full_path)  # Remove or comment out if not needed
                    # exit()  # Commented out to avoid premature termination
                    self.file_list.append({
                        "path": full_path,
                        "label": 0 
                    })

            root_dir = "<path_to_dataset>/data_c40/manipulated_sequences/NeuralTextures/c40/v1faces"
                    
            for directory in os.listdir(root_dir):
                dir_path = os.path.join(root_dir, directory)

                for file in os.listdir(dir_path):
                    full_path = os.path.join(dir_path, file)
                    # print(full_path)  # Remove or comment out if not needed
                    # exit()  # Commented out to avoid premature termination
                    self.file_list.append({
                        "path": full_path,
                        "label": 1
                    })

            
        if partition=="train":
            # ❶  RAW quality (uncompressed)
            _pair_real_fake(
                real_root = Path("<path_to_dataset>/original_sequences/youtube/raw/v1faces"),
                fake_root = Path("<path_to_dataset>/manipulated_sequences/NeuralTextures/raw/v1faces"),
                quality_tag = "raw",
                file_list   = self.file_list
            )

            # ❷  C40 quality (compressed / low-quality)
            _pair_real_fake(
                real_root = Path("<path_to_dataset>data_c40/original_sequences/youtube/c40/v1faces"),
                fake_root = Path("<path_to_dataset>data_c40/manipulated_sequences/NeuralTextures/c40/v1faces"),
                quality_tag = "c40",
                file_list   = self.file_list
            )
            
        print(f"Total files in {partition} are {len(self.file_list)}")
     


        # Initialize MFCC parameters
        self.n_mfcc = n_mfcc
        self.mfcc_max_len = 128  # Number of frames for MFCC (adjust as needed)

    def __len__(self):
        return len(self.file_list)



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
            data = self.file_list[index]
            file_path = data["path"]
            fake_path = data["fake_path"] if "fake_path" in data else None
            label = data["label"]
            # Generate text metadata
            filename = os.path.basename(file_path)
            image = Image.open(file_path)
            if fake_path is not None:
                fake_image = Image.open(fake_path)
            else:
                fake_image = Image.open(file_path)
                        
            target_size = (224, 224)

            image_tensor = TF.to_tensor(image)  # shape: [3, H, W]
            # Convert image to grayscale
            image_gray = TF.resize(TF.rgb_to_grayscale(image_tensor), target_size)


            image_aug = self.image_aug(image)
            fake_image_aug = self.image_aug(image)
            if self.is_train:
                ori_image  = self.image_aug_ori(image)
                fake_image = self.image_aug(fake_image)
            else:
                ori_image = self.image_no_aug(image)
            if fake_path is None:
                return  ori_image, image_aug, label, os.path.basename(file_path)
            else:
                return  ori_image,fake_image, image_aug, fake_image_aug, label, os.path.basename(file_path)
        except Exception as e:
            traceback.print_exc()
            print(f"Error loading {file_path}: {e}")
        
            return self.__getitem__((index + 1) % len(self.file_list))