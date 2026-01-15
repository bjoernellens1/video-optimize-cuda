# Video Optimize CUDA

An end-to-end video deblurring pipeline leveraging modern AI models (NAFNet) with CUDA acceleration. Optimized for efficient batch processing of high-definition videos with up to 16GB GPU VRAM.

## Features

- 🚀 **CUDA-Accelerated**: GPU-optimized processing for fast video deblurring
- 🎯 **State-of-the-Art AI**: Uses NAFNet (Nonlinear Activation Free Network) for high-quality deblurring
- 📦 **Batch Processing**: Process entire folders of videos end-to-end
- 💾 **Memory Efficient**: Tile-based processing and FP16 support for 16GB VRAM constraint
- ⚙️ **Configurable**: YAML-based configuration for easy customization
- 🎬 **Format Support**: Supports MP4, AVI, MOV, MKV, and more

## Architecture

The pipeline uses **NAFNet** (Nonlinear Activation Free Network), a state-of-the-art model for image restoration:
- Published in ECCV 2022
- Achieves excellent results without complex activation functions
- Efficient architecture suitable for video processing
- Paper: [Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676)

### Key Optimizations
- **Tile-based processing**: Processes large frames in smaller tiles to fit in VRAM
- **FP16 precision**: Half-precision floating point for 2x memory reduction
- **Dynamic resolution**: Automatic downscaling of oversized videos
- **Batch processing**: Efficient processing of multiple videos sequentially

## Requirements

- Python 3.8+
- CUDA-capable GPU (NVIDIA) with 8GB+ VRAM (16GB recommended)
- CUDA Toolkit 11.0+ (comes with PyTorch)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bjoernellens1/video-optimize-cuda.git
cd video-optimize-cuda
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Download pretrained NAFNet weights:
```bash
# For best results, download pretrained weights from the official NAFNet repository
# https://github.com/megvii-research/NAFNet
# Place the weights file in the project directory and update config.yaml
```

## Quick Start

1. **Create input directory** and add your videos:
```bash
mkdir -p input_videos
# Copy your video files to input_videos/
```

2. **Run the pipeline**:
```bash
python video_pipeline.py
```

3. **Find processed videos** in `output_videos/` directory:
```bash
ls output_videos/
```

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
# Input/Output paths
input_dir: "./input_videos"
output_dir: "./output_videos"

# Model configuration
model:
  name: "nafnet"
  checkpoint: null  # Path to pretrained weights

# Processing configuration
processing:
  batch_size: 4
  tile_size: 512  # Tile size for memory-efficient processing
  tile_overlap: 32
  max_resolution: [1920, 1080]  # Downscale if larger

# GPU configuration
gpu:
  device: "cuda:0"
  fp16: true  # Use half precision
  max_vram_gb: 16

# Output settings
output:
  codec: "mp4v"
  crf: 18
  preset: "slow"
  pixel_format: "yuv420p"
```

## Command Line Options

```bash
# Use custom config file
python video_pipeline.py --config my_config.yaml

# Override input/output directories
python video_pipeline.py --input /path/to/videos --output /path/to/output

# Use specific GPU device
python video_pipeline.py --device cuda:1

# Use CPU (slow, not recommended)
python video_pipeline.py --device cpu
```

## Usage Examples

### Example 1: Process single video folder
```bash
python video_pipeline.py --input ./my_videos --output ./deblurred_videos
```

### Example 2: High-quality 4K processing
```yaml
# config.yaml
processing:
  tile_size: 256  # Smaller tiles for 4K
  max_resolution: [3840, 2160]  # Allow 4K

gpu:
  fp16: true  # Essential for 4K on 16GB
```

### Example 3: Fast processing with lower quality
```yaml
processing:
  tile_size: 1024  # Larger tiles
  max_resolution: [1280, 720]  # HD only

output:
  preset: "fast"
  crf: 23  # Lower quality, smaller files
```

## Performance

Approximate processing speeds on RTX 3090 (24GB VRAM):

| Resolution | FP16 | Processing Speed |
|-----------|------|-----------------|
| 1080p     | Yes  | ~15-20 fps      |
| 1080p     | No   | ~8-10 fps       |
| 4K        | Yes  | ~4-6 fps        |
| 4K        | No   | ~2-3 fps        |

*Actual speed depends on video content complexity and hardware.*

## Memory Requirements

Estimated VRAM usage for different configurations:

| Resolution | Batch Size | FP16 | VRAM Usage |
|-----------|-----------|------|------------|
| 1920x1080 | 1         | Yes  | ~6 GB      |
| 1920x1080 | 1         | No   | ~12 GB     |
| 3840x2160 | 1         | Yes  | ~14 GB     |
| 1280x720  | 2         | Yes  | ~4 GB      |

## Project Structure

```
video-optimize-cuda/
├── video_pipeline.py      # Main pipeline script
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── models/
│   └── nafnet.py        # NAFNet model implementation
├── utils/
│   └── video_utils.py   # Video processing utilities
├── input_videos/        # Place input videos here
└── output_videos/       # Processed videos output here
```

## Pretrained Weights

For best results, download pretrained NAFNet weights:

1. Visit the [NAFNet repository](https://github.com/megvii-research/NAFNet)
2. Download weights for the deblurring task
3. Update `config.yaml`:
```yaml
model:
  checkpoint: "path/to/nafnet_weights.pth"
```

Without pretrained weights, the model will use random initialization (not recommended for production use).

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `tile_size` in config (e.g., 256 or 128)
- Reduce `max_resolution`
- Enable `fp16: true`
- Close other GPU applications

### Slow Processing
- Enable `fp16: true` if not already
- Increase `tile_size` if you have VRAM headroom
- Use faster codec preset: `preset: "fast"`
- Consider using lower resolution

### CUDA Not Available
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Poor Quality Results
- Download and use pretrained weights
- Reduce compression: set `crf: 15` or lower
- Avoid downscaling: increase `max_resolution`

## Advanced Usage

### Custom Model Configuration

You can adjust the NAFNet model size in `models/nafnet.py`:

```python
# Smaller model (faster, less VRAM)
model = create_nafnet_model(width=16, middle_blk_num=6)

# Larger model (better quality, more VRAM)
model = create_nafnet_model(width=64, middle_blk_num=16)
```

### Processing Specific Frames

Modify `video_pipeline.py` to process only certain frame ranges:

```python
# In process_video method, add frame range logic
if frame_count < start_frame or frame_count > end_frame:
    continue
```

## Citation

If you use this pipeline in your research, please cite NAFNet:

```bibtex
@inproceedings{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```

## License

This project is provided as-is for educational and research purposes. The NAFNet model implementation is based on the original paper. Please refer to the original NAFNet repository for model licensing.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Future Enhancements

- [ ] Support for other restoration tasks (denoising, super-resolution)
- [ ] Multi-GPU support for faster processing
- [ ] Real-time video stream processing
- [ ] Web UI for easy interaction
- [ ] Support for more AI models (Restormer, SwinIR, etc.)
- [ ] Video quality metrics (PSNR, SSIM)

## Contact

For questions or issues, please open an issue on GitHub.