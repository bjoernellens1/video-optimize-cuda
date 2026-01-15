#!/bin/bash
# Setup script for video-optimize-cuda

echo "=========================================="
echo "Video Optimize CUDA - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 not found. Please install pip3."
    exit 1
fi

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (recommended) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated."
fi

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# Create directories
echo ""
echo "Creating input/output directories..."
mkdir -p input_videos
mkdir -p output_videos
mkdir -p models

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Place your video files in 'input_videos/' directory"
echo "2. (Optional) Download pretrained NAFNet weights"
echo "3. Run: python video_pipeline.py"
echo ""
echo "For more information, see README.md"
echo ""
