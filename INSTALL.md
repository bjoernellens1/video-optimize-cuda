# Installation Guide

This guide provides detailed instructions for setting up the video deblurring pipeline.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060 or better recommended)
- **VRAM**: Minimum 8GB, 16GB recommended for HD/4K processing
- **RAM**: 16GB system RAM recommended
- **Storage**: Sufficient space for input and output videos

### Software Requirements
- **Operating System**: Linux, Windows, or macOS
- **Python**: 3.8 or higher
- **CUDA Toolkit**: 11.0 or higher (included with PyTorch)
- **FFmpeg**: For video encoding (usually included with OpenCV)

## Installation Steps

### 1. Install Python

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### macOS
```bash
brew install python3
```

#### Windows
Download and install from [python.org](https://www.python.org/downloads/)

### 2. Clone Repository

```bash
git clone https://github.com/bjoernellens1/video-optimize-cuda.git
cd video-optimize-cuda
```

### 3. Create Virtual Environment (Recommended)

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies

#### Option A: Automatic (Linux/macOS)
```bash
chmod +x setup.sh
./setup.sh
```

#### Option B: Manual
```bash
pip install -r requirements.txt
```

#### Option C: Install PyTorch with specific CUDA version
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install other dependencies
pip install opencv-python numpy Pillow tqdm timm einops imageio imageio-ffmpeg pyyaml
```

### 5. Verify Installation

```bash
python example.py
```

This will check your environment and create necessary directories.

## GPU Setup

### NVIDIA Driver Installation

#### Linux (Ubuntu/Debian)
```bash
# Add NVIDIA package repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install nvidia-driver-535

# Reboot
sudo reboot
```

#### Check Installation
```bash
nvidia-smi
```

### CUDA Toolkit (Optional)

PyTorch includes CUDA runtime, but you can install the full toolkit:

```bash
# Linux
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

### Verify CUDA in PyTorch

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

## Pretrained Weights Setup

For best results, download pretrained NAFNet weights:

### 1. Download Weights

Visit the [NAFNet repository](https://github.com/megvii-research/NAFNet) and download weights for:
- **Image Deblurring (GoPro)**: For motion blur
- **Image Deblurring (REDS)**: For video deblurring

Example download links:
```bash
# GoPro weights
wget https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-GoPro-width32.pth

# REDS weights  
wget https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-REDS-width64.pth
```

### 2. Update Configuration

Edit `config.yaml`:
```yaml
model:
  checkpoint: "NAFNet-GoPro-width32.pth"
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Solutions:**
1. Reduce tile size in config:
   ```yaml
   processing:
     tile_size: 256  # or 128
   ```

2. Enable FP16:
   ```yaml
   gpu:
     fp16: true
   ```

3. Reduce max resolution:
   ```yaml
   processing:
     max_resolution: [1280, 720]
   ```

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "CUDA not available"

**Possible causes:**
1. No NVIDIA GPU
2. NVIDIA driver not installed
3. PyTorch CPU version installed

**Solution:**
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "OpenCV video codec error"

**Solution (Linux):**
```bash
sudo apt install ffmpeg libsm6 libxext6
pip install opencv-python-headless
```

### Issue: Slow processing on CPU

**Solution:**
The pipeline requires a CUDA-capable GPU. Processing on CPU will be extremely slow (100x+ slower). Consider using a system with NVIDIA GPU or cloud GPU services.

## Cloud GPU Setup

If you don't have a local GPU, consider:

### Google Colab
```python
# In Colab notebook
!git clone https://github.com/bjoernellens1/video-optimize-cuda.git
%cd video-optimize-cuda
!pip install -r requirements.txt
# Upload videos via Colab interface
!python video_pipeline.py
```

### AWS EC2 with GPU
1. Launch EC2 instance with GPU (e.g., g4dn.xlarge)
2. Install NVIDIA drivers
3. Follow standard installation

### Paperspace Gradient
1. Create a GPU machine
2. Clone and run setup

## Performance Optimization

### For Maximum Speed
```yaml
processing:
  tile_size: 1024
  max_resolution: [1920, 1080]
gpu:
  fp16: true
output:
  preset: "ultrafast"
  crf: 23
```

### For Maximum Quality
```yaml
processing:
  tile_size: 512
  max_resolution: [3840, 2160]
gpu:
  fp16: false  # if you have VRAM
output:
  preset: "veryslow"
  crf: 15
```

## Next Steps

After successful installation:
1. Read the main [README.md](README.md) for usage instructions
2. Place test videos in `input_videos/`
3. Run `python video_pipeline.py`
4. Check results in `output_videos/`

## Getting Help

If you encounter issues:
1. Check this installation guide
2. Review troubleshooting section
3. Open an issue on GitHub with:
   - Error message
   - System information (GPU, CUDA version, Python version)
   - Configuration used
