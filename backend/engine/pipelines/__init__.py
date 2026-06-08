"""Image / video / audio assembly holders (registry-driven phased execution)."""

from .audio_pipeline import AudioPipeline
from .image_pipeline import ImagePipeline
from .music_pipeline import MusicPipeline
from .upscale_pipeline import UpscalePipeline
from .video_pipeline import VideoPipeline
from .video_upscale_pipeline import VideoUpscalePipeline

ImageUpscalePipeline = UpscalePipeline

__all__ = [
    "AudioPipeline",
    "ImagePipeline",
    "ImageUpscalePipeline",
    "MusicPipeline",
    "UpscalePipeline",
    "VideoPipeline",
    "VideoUpscalePipeline",
]
