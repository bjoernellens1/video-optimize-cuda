#!/usr/bin/env python3
"""
Quick Start Example
Simple script to test the video deblurring pipeline
"""

import os
import sys
import torch
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_pipeline import VideoDeblurPipeline


def check_environment():
    """Check if the environment is properly set up."""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            print(f"    VRAM: {vram_gb:.2f} GB")
    else:
        print("WARNING: CUDA not available. Processing will be very slow on CPU.")
        print("Consider installing PyTorch with CUDA support.")
    
    print("=" * 60)
    print()


def create_sample_directories():
    """Create sample input/output directories."""
    input_dir = Path("input_videos")
    output_dir = Path("output_videos")
    
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    return input_dir, output_dir


def main():
    """Run the example."""
    print("\n" + "=" * 60)
    print("Video Deblurring Pipeline - Quick Start")
    print("=" * 60 + "\n")
    
    # Check environment
    check_environment()
    
    # Create directories
    input_dir, output_dir = create_sample_directories()
    
    # Check for input videos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    input_videos = []
    for ext in video_extensions:
        input_videos.extend(list(input_dir.glob(f"*{ext}")))
    
    if not input_videos:
        print("📁 No videos found in input_videos/ directory")
        print("\nTo get started:")
        print("1. Place your video files in the 'input_videos/' directory")
        print("2. Run this script again: python example.py")
        print("\nSupported formats: .mp4, .avi, .mov, .mkv")
        print("=" * 60)
        return
    
    print(f"📹 Found {len(input_videos)} video(s) to process:")
    for video in input_videos:
        print(f"   - {video.name}")
    print()
    
    # Create and run pipeline
    try:
        print("🚀 Starting pipeline...")
        pipeline = VideoDeblurPipeline("config.yaml")
        pipeline.run()
        
        print("\n✅ Processing complete!")
        print(f"📂 Output videos saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
