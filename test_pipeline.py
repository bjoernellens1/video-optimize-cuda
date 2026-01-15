"""
Simple tests for the video deblurring pipeline
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.nafnet import create_nafnet_model
from utils.video_utils import preprocess_frame, postprocess_frame, estimate_vram_usage


def test_model_creation():
    """Test that model can be created."""
    print("Test: Model Creation")
    model = create_nafnet_model(width=16)
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    print(f"  ✓ Model created with {param_count/1e6:.2f}M parameters")


def test_model_forward():
    """Test that model can process input."""
    print("Test: Model Forward Pass")
    model = create_nafnet_model(width=16)
    model.eval()
    
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    
    assert y.shape == x.shape
    print(f"  ✓ Model forward pass successful: {x.shape} -> {y.shape}")


def test_preprocess_postprocess():
    """Test frame preprocessing and postprocessing."""
    print("Test: Frame Preprocessing/Postprocessing")
    
    # Create dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Preprocess
    device = torch.device('cpu')
    tensor = preprocess_frame(frame, device, fp16=False)
    
    assert tensor.shape == (1, 3, 480, 640)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    
    # Postprocess
    output_frame = postprocess_frame(tensor)
    
    assert output_frame.shape == frame.shape
    assert output_frame.dtype == np.uint8
    
    print(f"  ✓ Preprocessing/postprocessing successful")


def test_vram_estimation():
    """Test VRAM estimation function."""
    print("Test: VRAM Estimation")
    
    # Test different configurations
    configs = [
        (1080, 1920, 1, True, "1080p FP16"),
        (1080, 1920, 1, False, "1080p FP32"),
        (2160, 3840, 1, True, "4K FP16"),
    ]
    
    for height, width, batch, fp16, desc in configs:
        vram = estimate_vram_usage(height, width, batch, fp16)
        assert vram > 0
        print(f"  ✓ {desc}: {vram:.2f} GB")


def test_config_loading():
    """Test configuration loading."""
    print("Test: Configuration Loading")
    
    from video_pipeline import VideoDeblurPipeline
    
    # Create directories if needed
    os.makedirs('input_videos', exist_ok=True)
    os.makedirs('output_videos', exist_ok=True)
    
    pipeline = VideoDeblurPipeline('config.yaml')
    assert pipeline.config is not None
    assert 'input_dir' in pipeline.config
    assert 'output_dir' in pipeline.config
    
    print(f"  ✓ Configuration loaded successfully")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Pipeline Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_model_creation,
        test_model_forward,
        test_preprocess_postprocess,
        test_vram_estimation,
        test_config_loading,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
