"""
Video Deblurring Pipeline
End-to-end batch processing of videos using AI models with CUDA acceleration
"""

import os
import sys
import yaml
import torch
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.nafnet import create_nafnet_model
from utils.video_utils import (
    load_video, get_video_writer, preprocess_frame, postprocess_frame,
    tile_process, get_video_files, resize_if_needed, estimate_vram_usage
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoDeblurPipeline:
    """Video deblurring pipeline with CUDA optimization"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.device = torch.device(self.config['gpu']['device'] if torch.cuda.is_available() else 'cpu')
        self.fp16 = self.config['gpu']['fp16'] and torch.cuda.is_available()
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"FP16 mode: {self.fp16}")
        
        # Load model
        self.model = self.load_model()
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self.get_default_config()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'input_dir': './input_videos',
            'output_dir': './output_videos',
            'model': {
                'name': 'nafnet',
                'checkpoint': None
            },
            'processing': {
                'batch_size': 1,
                'tile_size': 512,
                'tile_overlap': 32,
                'max_resolution': [1920, 1080]
            },
            'gpu': {
                'device': 'cuda:0',
                'fp16': True,
                'max_vram_gb': 16
            },
            'output': {
                'codec': 'mp4v',
                'crf': 18,
                'preset': 'slow',
                'pixel_format': 'yuv420p'
            }
        }
    
    def load_model(self) -> torch.nn.Module:
        """Load and prepare the deblurring model."""
        logger.info("Loading NAFNet model...")
        
        # Create model
        model = create_nafnet_model(width=32)
        model = model.to(self.device)
        
        if self.fp16:
            model = model.half()
        
        model.eval()
        
        # Load checkpoint if provided
        checkpoint_path = self.config['model'].get('checkpoint')
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=False)
            logger.info("Checkpoint loaded successfully")
        else:
            logger.info("No checkpoint provided, using randomly initialized weights")
            logger.warning("For best results, download pretrained weights from NAFNet repository")
        
        return model
    
    def process_frame(self, frame):
        """
        Process a single frame through the model.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Processed frame as numpy array
        """
        # Resize if needed
        max_res = self.config['processing'].get('max_resolution')
        if max_res:
            frame, original_size = resize_if_needed(frame, tuple(max_res))
        
        # Preprocess
        input_tensor = preprocess_frame(frame, self.device, self.fp16)
        
        # Process with model
        with torch.no_grad():
            tile_size = self.config['processing'].get('tile_size')
            if tile_size:
                output_tensor = tile_process(
                    self.model, 
                    input_tensor,
                    tile_size=tile_size,
                    tile_overlap=self.config['processing'].get('tile_overlap', 32)
                )
            else:
                output_tensor = self.model(input_tensor)
        
        # Postprocess
        output_frame = postprocess_frame(output_tensor)
        
        return output_frame
    
    def process_video(self, video_path: Path, output_path: Path):
        """
        Process a single video file.
        
        Args:
            video_path: Input video path
            output_path: Output video path
        """
        logger.info(f"Processing: {video_path.name}")
        
        # Load video
        cap, metadata = load_video(str(video_path))
        
        logger.info(f"Video info: {metadata['width']}x{metadata['height']}, "
                   f"{metadata['fps']:.2f} fps, {metadata['frame_count']} frames")
        
        # Estimate VRAM usage
        vram_estimate = estimate_vram_usage(
            metadata['height'], 
            metadata['width'],
            1,  # Process one frame at a time for video
            self.fp16
        )
        logger.info(f"Estimated VRAM usage: {vram_estimate:.2f} GB")
        
        if vram_estimate > self.config['gpu']['max_vram_gb']:
            logger.warning(f"Estimated VRAM exceeds limit, processing may be slow or fail")
        
        # Setup video writer
        codec = self.config['output'].get('codec', 'mp4v')
        writer = get_video_writer(
            str(output_path),
            metadata['fps'],
            metadata['width'],
            metadata['height'],
            codec=codec
        )
        
        # Process frames
        frame_count = 0
        pbar = tqdm(total=metadata['frame_count'], desc="Processing frames")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Write to output
                writer.write(processed_frame)
                
                frame_count += 1
                pbar.update(1)
                
                # Clear CUDA cache periodically
                if frame_count % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        finally:
            pbar.close()
            cap.release()
            writer.release()
        
        logger.info(f"Completed: {video_path.name} -> {output_path.name}")
        logger.info(f"Processed {frame_count} frames")
    
    def run(self):
        """Run the pipeline on all videos in input directory."""
        logger.info("=" * 80)
        logger.info("Video Deblurring Pipeline Started")
        logger.info("=" * 80)
        
        # Get input videos
        input_dir = self.config['input_dir']
        video_files = get_video_files(input_dir)
        
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            logger.info("Please place video files in the input directory and try again.")
            return
        
        logger.info(f"Found {len(video_files)} video(s) to process")
        
        # Process each video
        for video_path in video_files:
            output_path = Path(self.config['output_dir']) / f"deblurred_{video_path.name}"
            
            try:
                self.process_video(video_path, output_path)
            except Exception as e:
                logger.error(f"Error processing {video_path.name}: {str(e)}", exc_info=True)
                continue
        
        logger.info("=" * 80)
        logger.info("Pipeline completed!")
        logger.info(f"Output videos saved to: {self.config['output_dir']}")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Video Deblurring Pipeline with CUDA acceleration"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Override input directory from config'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Override output directory from config'
    )
    parser.add_argument(
        '--device',
        type=str,
        help='Override CUDA device (e.g., cuda:0, cpu)'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = VideoDeblurPipeline(args.config)
    
    # Override config with command line args
    if args.input:
        pipeline.config['input_dir'] = args.input
    if args.output:
        pipeline.config['output_dir'] = args.output
    if args.device:
        pipeline.config['gpu']['device'] = args.device
        pipeline.device = torch.device(args.device)
    
    # Run pipeline
    pipeline.run()


if __name__ == "__main__":
    main()
