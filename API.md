# API Documentation

This document provides detailed API documentation for the video deblurring pipeline.

## Core Modules

### `video_pipeline.py`

The main pipeline module for end-to-end video processing.

#### Class: `VideoDeblurPipeline`

Main class for video deblurring operations.

**Constructor:**
```python
VideoDeblurPipeline(config_path: str = "config.yaml")
```

**Parameters:**
- `config_path` (str): Path to YAML configuration file

**Methods:**

##### `load_config(config_path: str) -> dict`
Load configuration from YAML file.

##### `load_model() -> torch.nn.Module`
Load and initialize the deblurring model.

**Returns:** Configured PyTorch model

##### `process_frame(frame: np.ndarray) -> np.ndarray`
Process a single video frame.

**Parameters:**
- `frame` (np.ndarray): Input frame in BGR format (H, W, 3)

**Returns:** Processed frame in BGR format

##### `process_video(video_path: Path, output_path: Path)`
Process a complete video file.

**Parameters:**
- `video_path` (Path): Input video file path
- `output_path` (Path): Output video file path

##### `run()`
Run the pipeline on all videos in the input directory.

**Example Usage:**
```python
from video_pipeline import VideoDeblurPipeline

# Create pipeline
pipeline = VideoDeblurPipeline("config.yaml")

# Override settings
pipeline.config['processing']['tile_size'] = 256

# Run on all videos
pipeline.run()
```

### `models/nafnet.py`

NAFNet model implementation for image deblurring.

#### Class: `NAFNet`

Nonlinear Activation Free Network for image restoration.

**Constructor:**
```python
NAFNet(
    img_channel: int = 3,
    width: int = 32,
    middle_blk_num: int = 12,
    enc_blk_nums: List[int] = [2, 2, 4, 8],
    dec_blk_nums: List[int] = [2, 2, 2, 2]
)
```

**Parameters:**
- `img_channel` (int): Number of image channels (default: 3)
- `width` (int): Base channel width (default: 32)
- `middle_blk_num` (int): Number of middle blocks (default: 12)
- `enc_blk_nums` (List[int]): Blocks per encoder stage
- `dec_blk_nums` (List[int]): Blocks per decoder stage

**Methods:**

##### `forward(inp: torch.Tensor) -> torch.Tensor`
Process input image tensor.

**Parameters:**
- `inp` (torch.Tensor): Input tensor (B, C, H, W)

**Returns:** Deblurred image tensor (B, C, H, W)

#### Function: `create_nafnet_model`

Factory function to create NAFNet models.

```python
create_nafnet_model(
    width: int = 32,
    middle_blk_num: int = 12,
    enc_blk_nums: List[int] = [2, 2, 4, 8],
    dec_blk_nums: List[int] = [2, 2, 2, 2]
) -> NAFNet
```

**Example Usage:**
```python
from models.nafnet import create_nafnet_model
import torch

# Create model
model = create_nafnet_model(width=32)
model.eval()

# Process image
x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    y = model(x)
```

### `utils/video_utils.py`

Utility functions for video processing.

#### Functions:

##### `load_video(video_path: str) -> Tuple[cv2.VideoCapture, dict]`
Load video and extract metadata.

**Parameters:**
- `video_path` (str): Path to video file

**Returns:** Tuple of (VideoCapture object, metadata dict)

**Metadata keys:**
- `fps`: Frames per second
- `frame_count`: Total number of frames
- `width`: Frame width in pixels
- `height`: Frame height in pixels
- `fourcc`: Video codec

**Example:**
```python
from utils.video_utils import load_video

cap, metadata = load_video("input.mp4")
print(f"Video: {metadata['width']}x{metadata['height']} @ {metadata['fps']} fps")
```

##### `get_video_writer(output_path: str, fps: float, width: int, height: int, codec: str) -> cv2.VideoWriter`
Create video writer for output.

##### `preprocess_frame(frame: np.ndarray, device: torch.device, fp16: bool = False) -> torch.Tensor`
Convert frame to tensor and normalize.

**Parameters:**
- `frame` (np.ndarray): Input frame (H, W, C) in BGR
- `device` (torch.device): Target device
- `fp16` (bool): Use half precision

**Returns:** Preprocessed tensor (1, C, H, W)

##### `postprocess_frame(tensor: torch.Tensor) -> np.ndarray`
Convert tensor back to frame.

**Parameters:**
- `tensor` (torch.Tensor): Output tensor (1, C, H, W)

**Returns:** Frame as numpy array (H, W, C) in BGR

##### `tile_process(model: torch.nn.Module, img: torch.Tensor, tile_size: int = 512, tile_overlap: int = 32) -> torch.Tensor`
Process large images in tiles to save VRAM.

**Parameters:**
- `model` (torch.nn.Module): Deblurring model
- `img` (torch.Tensor): Input tensor (1, C, H, W)
- `tile_size` (int): Size of each tile
- `tile_overlap` (int): Overlap between tiles

**Returns:** Processed tensor

**Example:**
```python
from utils.video_utils import tile_process

# Process large image in tiles
output = tile_process(model, large_image, tile_size=512, tile_overlap=32)
```

##### `get_video_files(input_dir: str, extensions: List[str] = None) -> List[Path]`
Get all video files from input directory.

##### `resize_if_needed(frame: np.ndarray, max_resolution: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Tuple[int, int]]`
Resize frame if it exceeds max resolution.

##### `estimate_vram_usage(height: int, width: int, batch_size: int, fp16: bool = False) -> float`
Estimate VRAM usage in GB.

**Parameters:**
- `height` (int): Frame height
- `width` (int): Frame width
- `batch_size` (int): Number of frames to process together
- `fp16` (bool): Using half precision

**Returns:** Estimated VRAM in GB

**Example:**
```python
from utils.video_utils import estimate_vram_usage

# Check if configuration will fit in VRAM
vram_needed = estimate_vram_usage(1080, 1920, batch_size=1, fp16=True)
if vram_needed > 16:
    print("Configuration may not fit in 16GB VRAM")
```

## Configuration Schema

The `config.yaml` file controls all pipeline parameters.

### Full Configuration Example:

```yaml
# Input/Output paths
input_dir: "./input_videos"
output_dir: "./output_videos"

# Model configuration
model:
  name: "nafnet"
  checkpoint: null  # Path to pretrained weights, or null

# Processing configuration
processing:
  batch_size: 4  # Frames to process at once (not used for video)
  tile_size: 512  # Tile size for memory-efficient processing
  tile_overlap: 32  # Overlap between tiles
  max_resolution: [1920, 1080]  # Maximum [width, height]

# GPU configuration
gpu:
  device: "cuda:0"  # CUDA device or "cpu"
  fp16: true  # Use half precision
  max_vram_gb: 16  # Maximum VRAM in GB

# Video encoding settings
output:
  codec: "mp4v"  # Video codec fourcc
  crf: 18  # Quality (0-51, lower = better)
  preset: "slow"  # Encoding speed
  pixel_format: "yuv420p"  # Pixel format
```

### Configuration Parameters:

#### `input_dir` (string)
Directory containing input videos to process.

#### `output_dir` (string)
Directory where processed videos will be saved.

#### `model.name` (string)
Model architecture name. Currently supports: `"nafnet"`

#### `model.checkpoint` (string | null)
Path to pretrained model weights. If null, uses random initialization.

#### `processing.batch_size` (integer)
Number of frames to process together (currently not used for videos).

#### `processing.tile_size` (integer | null)
Size of tiles for memory-efficient processing. Smaller values use less VRAM but may be slower. Set to null to disable tiling.

**Recommendations:**
- 16GB VRAM, 1080p: 512-1024
- 16GB VRAM, 4K: 256-512
- 8GB VRAM: 256

#### `processing.tile_overlap` (integer)
Overlap between tiles in pixels to avoid seam artifacts.

#### `processing.max_resolution` (array[width, height])
Maximum resolution. Videos larger than this will be downscaled.

#### `gpu.device` (string)
PyTorch device string. Examples: `"cuda:0"`, `"cuda:1"`, `"cpu"`

#### `gpu.fp16` (boolean)
Enable half-precision (FP16) computation for ~2x memory reduction.

#### `gpu.max_vram_gb` (number)
Maximum VRAM in GB. Used for warnings only, doesn't limit actual usage.

#### `output.codec` (string)
Video codec fourcc code. Examples: `"mp4v"`, `"avc1"`, `"XVID"`

#### `output.crf` (integer)
Constant Rate Factor for quality (H.264/H.265). Range: 0-51
- 0: Lossless
- 18: Visually lossless
- 23: Default
- 28: Lower quality

#### `output.preset` (string)
Encoding preset. Options: `"ultrafast"`, `"fast"`, `"medium"`, `"slow"`, `"veryslow"`

## Command-Line Interface

### `video_pipeline.py`

```bash
python video_pipeline.py [OPTIONS]
```

**Options:**
- `--config PATH`: Path to configuration file (default: `config.yaml`)
- `--input PATH`: Override input directory from config
- `--output PATH`: Override output directory from config
- `--device DEVICE`: Override CUDA device (e.g., `cuda:0`, `cpu`)

**Examples:**

```bash
# Use default config
python video_pipeline.py

# Use custom config
python video_pipeline.py --config my_config.yaml

# Override input/output
python video_pipeline.py --input /data/videos --output /data/deblurred

# Use specific GPU
python video_pipeline.py --device cuda:1

# Use CPU (slow)
python video_pipeline.py --device cpu
```

### `example.py`

Quick start script with environment checking.

```bash
python example.py
```

### `test_pipeline.py`

Run tests to verify installation.

```bash
python test_pipeline.py
```

### `download_weights.py`

Interactive script to download pretrained weights.

```bash
python download_weights.py
```

## Integration Examples

### Python Script Integration

```python
import sys
sys.path.insert(0, '/path/to/video-optimize-cuda')

from video_pipeline import VideoDeblurPipeline
from pathlib import Path

# Create custom pipeline
pipeline = VideoDeblurPipeline("config.yaml")

# Process specific video
input_video = Path("my_video.mp4")
output_video = Path("deblurred_video.mp4")
pipeline.process_video(input_video, output_video)
```

### Jupyter Notebook Integration

```python
import sys
sys.path.insert(0, '/path/to/video-optimize-cuda')

from video_pipeline import VideoDeblurPipeline
from IPython.display import Video

# Create and run pipeline
pipeline = VideoDeblurPipeline()
pipeline.run()

# Display result
Video("output_videos/deblurred_video.mp4", width=800)
```

### Custom Model Integration

```python
from models.nafnet import NAFNet
import torch

# Create custom model configuration
custom_model = NAFNet(
    img_channel=3,
    width=64,  # Larger model
    middle_blk_num=16,
    enc_blk_nums=[2, 2, 4, 8],
    dec_blk_nums=[2, 2, 2, 2]
)

# Load custom weights
checkpoint = torch.load("custom_weights.pth")
custom_model.load_state_dict(checkpoint)

# Use in pipeline
pipeline = VideoDeblurPipeline()
pipeline.model = custom_model
pipeline.run()
```

## Performance Tuning

### Memory Optimization

To reduce VRAM usage:
1. Enable FP16: `gpu.fp16: true`
2. Reduce tile size: `processing.tile_size: 256`
3. Reduce resolution: `processing.max_resolution: [1280, 720]`
4. Use smaller model: `width=16` or `width=32`

### Speed Optimization

To increase processing speed:
1. Increase tile size if VRAM allows
2. Use faster encoding preset: `output.preset: "fast"`
3. Reduce output quality: `output.crf: 23`
4. Disable unnecessary processing

### Quality Optimization

For best quality:
1. Use pretrained weights
2. Disable FP16 if VRAM allows: `gpu.fp16: false`
3. Increase tile size: `processing.tile_size: 1024`
4. Use higher quality encoding: `output.crf: 15`
5. Use larger model: `width=64`

## Error Handling

### Common Errors:

**CUDA Out of Memory:**
```python
RuntimeError: CUDA out of memory
```
Solution: Reduce `tile_size` or enable `fp16`

**Video Codec Error:**
```python
cv2.error: OpenCV(4.x.x) error: (-2:Unspecified error)
```
Solution: Install ffmpeg or use different codec

**Module Not Found:**
```python
ModuleNotFoundError: No module named 'torch'
```
Solution: `pip install -r requirements.txt`

## License & Citation

See main README.md for license information and citation guidelines.
