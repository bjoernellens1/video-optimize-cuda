#!/usr/bin/env python3
"""
Download pretrained NAFNet weights for video deblurring.
"""

import os
import sys
import urllib.request
from pathlib import Path


WEIGHTS_INFO = {
    'nafnet-gopro-width32': {
        'url': 'https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-GoPro-width32.pth',
        'filename': 'NAFNet-GoPro-width32.pth',
        'size_mb': 68,
        'description': 'NAFNet trained on GoPro dataset (motion blur) - Recommended for most videos',
    },
    'nafnet-gopro-width64': {
        'url': 'https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-GoPro-width64.pth',
        'filename': 'NAFNet-GoPro-width64.pth',
        'size_mb': 272,
        'description': 'NAFNet-64 trained on GoPro - Higher quality but requires more VRAM',
    },
}


def download_file(url, filename, description, size_mb):
    """Download a file with progress bar."""
    print(f"\nDownloading: {description}")
    print(f"URL: {url}")
    print(f"Saving to: {filename}")
    print(f"Size: ~{size_mb} MB (estimated)")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, filename, report_progress)
        print("\n✓ Download complete!")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {str(e)}")
        return False


def main():
    print("=" * 70)
    print("NAFNet Pretrained Weights Downloader")
    print("=" * 70)
    
    print("\nAvailable models:")
    for i, (key, info) in enumerate(WEIGHTS_INFO.items(), 1):
        print(f"\n{i}. {info['filename']}")
        print(f"   {info['description']}")
        print(f"   Size: ~{info['size_mb']} MB")
    
    print("\nNote: These weights are from the official NAFNet repository:")
    print("https://github.com/megvii-research/NAFNet")
    print("\nFor best results, use these pretrained weights instead of random initialization.")
    
    # Get user choice
    print("\n" + "=" * 70)
    choice = input("Select model to download (1-2, or 'q' to quit): ").strip()
    
    if choice.lower() == 'q':
        print("Exiting...")
        return
    
    try:
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(WEIGHTS_INFO):
            print("Invalid choice!")
            return
    except ValueError:
        print("Invalid input!")
        return
    
    # Get selected model
    model_key = list(WEIGHTS_INFO.keys())[choice_idx]
    model_info = WEIGHTS_INFO[model_key]
    
    # Check if file exists
    if os.path.exists(model_info['filename']):
        overwrite = input(f"\n{model_info['filename']} already exists. Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("Skipping download.")
            return
    
    # Download
    success = download_file(
        model_info['url'],
        model_info['filename'],
        model_info['description'],
        model_info['size_mb']
    )
    
    if success:
        print("\n" + "=" * 70)
        print("Next steps:")
        print(f"1. Update config.yaml to use the downloaded weights:")
        print(f"   model:")
        print(f"     checkpoint: \"{model_info['filename']}\"")
        print("\n2. Run the pipeline:")
        print("   python video_pipeline.py")
        print("=" * 70)
    else:
        print("\nIf automatic download fails, you can manually download from:")
        print(model_info['url'])
        print(f"and save it as: {model_info['filename']}")


if __name__ == "__main__":
    main()
