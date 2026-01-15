"""Utility functions for video processing and model loading"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_video(video_path: str) -> Tuple[cv2.VideoCapture, dict]:
    """
    Load video and extract metadata.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (VideoCapture object, metadata dict)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    metadata = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC))
    }
    
    return cap, metadata


def get_video_writer(output_path: str, fps: float, width: int, height: int,
                     codec: str = 'mp4v') -> cv2.VideoWriter:
    """
    Create video writer for output.
    
    Args:
        output_path: Output video path
        fps: Frames per second
        width: Frame width
        height: Frame height
        codec: Video codec fourcc code
        
    Returns:
        VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        raise ValueError(f"Cannot create video writer: {output_path}")
    
    return writer


def preprocess_frame(frame: np.ndarray, device: torch.device, fp16: bool = False) -> torch.Tensor:
    """
    Convert frame to tensor and normalize.
    
    Args:
        frame: Input frame (H, W, C) in BGR format
        device: Target device
        fp16: Use half precision
        
    Returns:
        Preprocessed tensor (1, C, H, W)
    """
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and normalize to [0, 1]
    frame = torch.from_numpy(frame).float() / 255.0
    
    # Rearrange to (C, H, W) and add batch dimension
    frame = frame.permute(2, 0, 1).unsqueeze(0)
    
    # Move to device
    frame = frame.to(device)
    
    if fp16:
        frame = frame.half()
    
    return frame


def postprocess_frame(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor back to frame.
    
    Args:
        tensor: Output tensor (1, C, H, W)
        
    Returns:
        Frame as numpy array (H, W, C) in BGR format
    """
    # Remove batch dimension and move to CPU
    frame = tensor.squeeze(0).cpu().float()
    
    # Clamp to [0, 1] and convert to [0, 255]
    frame = torch.clamp(frame, 0, 1) * 255.0
    
    # Rearrange to (H, W, C)
    frame = frame.permute(1, 2, 0).numpy().astype(np.uint8)
    
    # Convert RGB to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    return frame


def tile_process(model: torch.nn.Module, img: torch.Tensor, 
                tile_size: int = 512, tile_overlap: int = 32) -> torch.Tensor:
    """
    Process large images in tiles to save VRAM.
    
    Args:
        model: Deblurring model
        img: Input tensor (1, C, H, W)
        tile_size: Size of each tile
        tile_overlap: Overlap between tiles
        
    Returns:
        Processed tensor
    """
    b, c, h, w = img.shape
    
    # If image is smaller than tile size, process directly
    if h <= tile_size and w <= tile_size:
        return model(img)
    
    # Calculate number of tiles
    stride = tile_size - tile_overlap
    h_tiles = (h - tile_overlap) // stride + 1
    w_tiles = (w - tile_overlap) // stride + 1
    
    # Output tensor
    output = torch.zeros_like(img)
    count = torch.zeros_like(img)
    
    for i in range(h_tiles):
        for j in range(w_tiles):
            # Calculate tile boundaries
            h_start = i * stride
            h_end = min(h_start + tile_size, h)
            w_start = j * stride
            w_end = min(w_start + tile_size, w)
            
            # Adjust start if we're at the edge
            if h_end == h:
                h_start = max(0, h - tile_size)
            if w_end == w:
                w_start = max(0, w - tile_size)
            
            h_end = min(h_start + tile_size, h)
            w_end = min(w_start + tile_size, w)
            
            # Extract and process tile
            tile = img[:, :, h_start:h_end, w_start:w_end]
            
            with torch.no_grad():
                processed_tile = model(tile)
            
            # Add to output with blending
            output[:, :, h_start:h_end, w_start:w_end] += processed_tile
            count[:, :, h_start:h_end, w_start:w_end] += 1
    
    # Average overlapping regions
    output = output / count
    
    return output


def get_video_files(input_dir: str, extensions: List[str] = None) -> List[Path]:
    """
    Get all video files from input directory.
    
    Args:
        input_dir: Input directory path
        extensions: List of video extensions to look for
        
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    video_files = []
    for ext in extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    return sorted(video_files)


def resize_if_needed(frame: np.ndarray, max_resolution: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize frame if it exceeds max resolution.
    
    Args:
        frame: Input frame
        max_resolution: Maximum (width, height)
        
    Returns:
        Tuple of (resized frame, original size)
    """
    h, w = frame.shape[:2]
    original_size = (w, h)
    
    if max_resolution is None:
        return frame, original_size
    
    max_w, max_h = max_resolution
    
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Make dimensions multiple of 8 for better model compatibility
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized frame from {w}x{h} to {new_w}x{new_h}")
    
    return frame, original_size


def estimate_vram_usage(height: int, width: int, batch_size: int, fp16: bool = False) -> float:
    """
    Estimate VRAM usage in GB.
    
    Args:
        height: Frame height
        width: Frame width
        batch_size: Number of frames to process together
        fp16: Using half precision
        
    Returns:
        Estimated VRAM in GB
    """
    bytes_per_element = 2 if fp16 else 4
    
    # Input frames
    input_size = batch_size * 3 * height * width * bytes_per_element
    
    # Model parameters (approximate for NAFNet-32)
    model_params = 68e6 * bytes_per_element
    
    # Activations (rough estimate, 10x input)
    activation_size = input_size * 10
    
    # Output frames
    output_size = input_size
    
    total_bytes = input_size + model_params + activation_size + output_size
    total_gb = total_bytes / (1024 ** 3)
    
    # Add 20% safety margin
    return total_gb * 1.2


if __name__ == "__main__":
    # Test VRAM estimation
    print("VRAM estimates for different configurations:")
    for res in [(1920, 1080), (1280, 720), (3840, 2160)]:
        for bs in [1, 2, 4]:
            vram_fp32 = estimate_vram_usage(res[1], res[0], bs, fp16=False)
            vram_fp16 = estimate_vram_usage(res[1], res[0], bs, fp16=True)
            print(f"{res[0]}x{res[1]}, batch={bs}: FP32={vram_fp32:.2f}GB, FP16={vram_fp16:.2f}GB")
